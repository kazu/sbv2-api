use axum::{
    extract::State,
    http::header::CONTENT_TYPE,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use sbv2_core::tts::{SynthesizeOptions, TTSModelHolder};
use serde::{Deserialize, Serialize};
use std::env;
use std::sync::Arc;
use tokio::fs;
use tokio::sync::Mutex;
use utoipa::{OpenApi, ToSchema};
use utoipa_scalar::{Scalar, Servable};

mod error;
use crate::error::AppResult;

#[derive(OpenApi)]
#[openapi(
    paths(models, synthesize, g2p),
    components(schemas(SynthesizeRequest, G2pRequest, G2pToken, G2pLine))
)]
struct ApiDoc;

#[utoipa::path(
    get,
    path = "/models",
    responses(
        (status = 200, description = "Return model list", body = Vec<String>),
    )
)]
async fn models(State(state): State<AppState>) -> AppResult<impl IntoResponse> {
    Ok(Json(state.tts_model.lock().await.models()))
}

fn sdp_default() -> f32 {
    0.0
}

fn length_default() -> f32 {
    1.0
}

fn style_id_default() -> i32 {
    0
}

fn speaker_id_default() -> i64 {
    0
}

#[derive(Deserialize, ToSchema)]
struct SynthesizeRequest {
    text: String,
    ident: String,
    #[serde(default = "sdp_default")]
    #[schema(example = 0.0_f32)]
    sdp_ratio: f32,
    #[serde(default = "length_default")]
    #[schema(example = 1.0_f32)]
    length_scale: f32,
    #[serde(default = "style_id_default")]
    #[schema(example = 0_i32)]
    style_id: i32,
    #[serde(default = "speaker_id_default")]
    #[schema(example = 0_i64)]
    speaker_id: i64,
}

#[utoipa::path(
    post,
    path = "/synthesize",
    request_body = SynthesizeRequest,
    responses(
        (status = 200, description = "Return audio/wav", body = Vec<u8>, content_type = "audio/wav")
    )
)]
async fn synthesize(
    State(state): State<AppState>,
    Json(SynthesizeRequest {
        text,
        ident,
        sdp_ratio,
        length_scale,
        style_id,
        speaker_id,
    }): Json<SynthesizeRequest>,
) -> AppResult<impl IntoResponse> {
    log::debug!("processing request: text={text}, ident={ident}, sdp_ratio={sdp_ratio}, length_scale={length_scale}");
    let buffer = {
        let mut tts_model = state.tts_model.lock().await;
        tts_model.easy_synthesize(
            &ident,
            &text,
            style_id,
            speaker_id,
            SynthesizeOptions {
                sdp_ratio,
                length_scale,
                ..Default::default()
            },
        )?
    };
    Ok(([(CONTENT_TYPE, "audio/wav")], buffer))
}

#[derive(Deserialize, ToSchema)]
struct G2pRequest {
    /// Text to grapheme-to-phoneme. May contain `\n`; each line is g2p'd separately
    /// (mirroring how synthesis splits on `\n`) so context-dependent readings are kept.
    text: String,
}

#[derive(Serialize, ToSchema)]
struct G2pToken {
    /// Token surface as segmented by the jpreprocess frontend.
    surface: String,
    /// Katakana pronunciation (NJD `pron`) SBV2 actually speaks for this surface.
    yomi: String,
}

#[derive(Serialize, ToSchema)]
struct G2pLine {
    /// The input line these tokens came from.
    text: String,
    /// Per-token `(surface, yomi)` in order. Empty if the frontend could not parse the line.
    tokens: Vec<G2pToken>,
}

/// Return SBV2's own per-token reading of `text` (CPU-only frontend, no synthesis).
///
/// This exposes exactly the readings synthesis uses, so callers can diff them against a
/// trusted source (author ruby / LLM) and correct only the words SBV2 misreads. It never
/// runs the ONNX model, so it is far lighter than `/synthesize`.
#[utoipa::path(
    post,
    path = "/g2p",
    request_body = G2pRequest,
    responses(
        (status = 200, description = "Per-line, per-token surface/yomi (katakana)", body = Vec<G2pLine>)
    )
)]
async fn g2p(
    State(state): State<AppState>,
    Json(G2pRequest { text }): Json<G2pRequest>,
) -> AppResult<impl IntoResponse> {
    let holder = state.tts_model.lock().await;
    let mut lines = Vec::new();
    for line in text.split('\n') {
        if line.is_empty() {
            continue;
        }
        // Best-effort: a line the frontend rejects yields empty tokens rather than a 500,
        // so one bad line never fails the whole (per-episode) request.
        let tokens = match holder.jtalk.process_text(line) {
            Ok(proc) => proc
                .tokens()
                .into_iter()
                .map(|(surface, yomi)| G2pToken { surface, yomi })
                .collect(),
            Err(e) => {
                log::debug!("g2p: skipping unparseable line {line:?}: {e}");
                Vec::new()
            }
        };
        lines.push(G2pLine {
            text: line.to_string(),
            tokens,
        });
    }
    Ok(Json(lines))
}

#[derive(Clone)]
struct AppState {
    tts_model: Arc<Mutex<TTSModelHolder>>,
}

impl AppState {
    pub async fn new() -> anyhow::Result<Self> {
        let mut tts_model = TTSModelHolder::new(
            &fs::read(env::var("BERT_MODEL_PATH")?).await?,
            &fs::read(env::var("TOKENIZER_PATH")?).await?,
            env::var("HOLDER_MAX_LOADED_MODElS")
                .ok()
                .and_then(|x| x.parse().ok()),
        )?;
        let models = env::var("MODELS_PATH").unwrap_or("models".to_string());
        let mut f = fs::read_dir(&models).await?;
        let mut entries = vec![];
        while let Ok(Some(e)) = f.next_entry().await {
            let name = e.file_name().to_string_lossy().to_string();
            if name.ends_with(".onnx") && name.starts_with("model_") {
                let name_len = name.len();
                let name = name.chars();
                entries.push(
                    name.collect::<Vec<_>>()[6..name_len - 5]
                        .iter()
                        .collect::<String>(),
                );
            } else if name.ends_with(".sbv2") {
                let entry = &name[..name.len() - 5];
                log::info!("Try loading: {entry}");
                if let Err(e) =
                    tts_model.load_sbv2file_path(entry, format!("{models}/{entry}.sbv2"))
                {
                    log::warn!("Error loading {entry}: {e}");
                };
                log::info!("Loaded: {entry}");
            } else if name.ends_with(".aivmx") {
                let entry = &name[..name.len() - 6];
                log::info!("Try loading: {entry}");
                if let Err(e) =
                    tts_model.load_aivmx_path(entry, format!("{models}/{entry}.aivmx"))
                {
                    log::error!("Error loading {entry}: {e}");
                }
                log::info!("Loaded: {entry}");
            }
        }
        for entry in entries {
            log::info!("Try loading: {entry}");
            let style_vectors_bytes =
                match fs::read(format!("{models}/style_vectors_{entry}.json")).await {
                    Ok(b) => b,
                    Err(e) => {
                        log::warn!("Error loading style_vectors_bytes from file {entry}: {e}");
                        continue;
                    }
                };
            let vits2_bytes = match fs::read(format!("{models}/model_{entry}.onnx")).await {
                Ok(b) => b,
                Err(e) => {
                    log::warn!("Error loading vits2_bytes from file {entry}: {e}");
                    continue;
                }
            };
            if let Err(e) = tts_model.load(&entry, style_vectors_bytes, vits2_bytes) {
                log::warn!("Error loading {entry}: {e}");
            };
            log::info!("Loaded: {entry}");
        }
        Ok(Self {
            tts_model: Arc::new(Mutex::new(tts_model)),
        })
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv_override().ok();
    env_logger::init();
    let app = Router::new()
        .route("/", get(|| async { "Hello, World!" }))
        .route("/synthesize", post(synthesize))
        .route("/g2p", post(g2p))
        .route("/models", get(models))
        .with_state(AppState::new().await?)
        .merge(Scalar::with_url("/docs", ApiDoc::openapi()));
    let addr = env::var("ADDR").unwrap_or("0.0.0.0:3000".to_string());
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    log::info!("Listening on {addr}");
    axum::serve(listener, app).await?;

    Ok(())
}
