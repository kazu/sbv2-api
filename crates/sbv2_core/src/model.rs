use crate::error::Result;
use ndarray::{array, Array1, Array2, Array3, Axis, Ix3};
use memmap2::Mmap;
use ort::session::{
    builder::{GraphOptimizationLevel, SessionBuilder},
    Session,
};
use std::path::Path;
use std::sync::OnceLock;

pub fn load_model<P: AsRef<[u8]>>(model_file: P, bert: bool) -> Result<Session> {
    Ok(session_builder(bert)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_memory(model_file.as_ref())?)
}

/// Same as `load_model`, also writing the optimized graph to `optimized` for
/// `load_optimized_model`: a `.ort` path gets the ORT format, any other path gets ONNX
/// with its prepacked weights in `<optimized>.data`.
pub fn load_model_saving_optimized<P: AsRef<[u8]>>(
    model_file: P,
    bert: bool,
    optimized: &Path,
) -> Result<Session> {
    let mut builder = prepack_knob(session_builder(bert)?)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_optimized_model_path(optimized)?;
    if !is_ort_format(optimized) {
        let mut data = optimized.file_name().unwrap_or_default().to_os_string();
        data.push(".data");
        builder = builder
            .with_config_entry(
                "session.optimized_model_external_initializers_file_name",
                data.to_string_lossy(),
            )?
            .with_config_entry("session.save_external_prepacked_constant_initializers", "1")?;
    }
    Ok(builder.commit_from_memory(model_file.as_ref())?)
}

/// A voice session and the mapped file its graph may point into. The field order drops
/// the session before the map.
pub struct Vits2 {
    pub session: Session,
    _map: Option<Mmap>,
}

impl std::ops::Deref for Vits2 {
    type Target = Session;
    fn deref(&self) -> &Session {
        &self.session
    }
}

impl std::ops::DerefMut for Vits2 {
    fn deref_mut(&mut self) -> &mut Session {
        &mut self.session
    }
}

impl From<Session> for Vits2 {
    fn from(session: Session) -> Self {
        Vits2 {
            session,
            _map: None,
        }
    }
}

/// Load a graph written by `load_model_saving_optimized` without optimizing it again.
/// A `.ort` file is mapped and used in place instead of being copied.
pub fn load_optimized_model(optimized: &Path, bert: bool) -> Result<Vits2> {
    let builder =
        prepack_knob(session_builder(bert)?)?.with_optimization_level(GraphOptimizationLevel::Disable)?;
    if is_ort_format(optimized) {
        // SAFETY: load_model_saving_optimized is the only writer and only writes a path
        // that does not exist yet.
        let t0 = std::time::Instant::now();
        let map = unsafe {
            memmap2::MmapOptions::new()
                .populate()
                .map(&std::fs::File::open(optimized)?)?
        };
        let t1 = std::time::Instant::now();
        let session = builder
            .with_config_entry("session.use_ort_model_bytes_directly", "1")?
            .with_config_entry("session.use_ort_model_bytes_for_initializers", "1")?
            .commit_from_memory(&map)?;
        log::debug!("timing map={:?} session={:?}", t1 - t0, t1.elapsed());
        Ok(Vits2 {
            session,
            _map: Some(map),
        })
    } else {
        Ok(builder.commit_from_file(optimized)?.into())
    }
}

fn is_ort_format(path: &Path) -> bool {
    path.extension().is_some_and(|e| e == "ort")
}

/// Experiment: SBV2_ORT_DISABLE_PREPACK leaves weights unpacked.
fn prepack_knob(builder: SessionBuilder) -> Result<SessionBuilder> {
    if std::env::var_os("SBV2_ORT_DISABLE_PREPACK").is_some() {
        Ok(builder.with_config_entry("session.disable_prepacking", "1")?)
    } else {
        Ok(builder)
    }
}

/// Experiment: SBV2_ORT_GLOBAL_POOL makes every session share one thread pool.
fn global_pool() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("SBV2_ORT_GLOBAL_POOL").is_some();
        if on {
            let n = num_cpus::get_physical();
            let pool = ort::environment::GlobalThreadPoolOptions::default()
                .with_intra_threads(n)
                .and_then(|p| p.with_inter_threads(n))
                .expect("thread pool options");
            ort::init().with_global_thread_pool(pool).commit();
        }
        on
    })
}

#[allow(clippy::vec_init_then_push, unused_variables)]
fn session_builder(bert: bool) -> Result<SessionBuilder> {
    let mut exp = Vec::new();
    #[cfg(feature = "tensorrt")]
    {
        if bert {
            exp.push(
                ort::execution_providers::TensorRTExecutionProvider::default()
                    .with_fp16(true)
                    .with_profile_min_shapes("input_ids:1x1,attention_mask:1x1")
                    .with_profile_max_shapes("input_ids:1x100,attention_mask:1x100")
                    .with_profile_opt_shapes("input_ids:1x25,attention_mask:1x25")
                    .build(),
            );
        }
    }
    #[cfg(feature = "cuda")]
    {
        #[allow(unused_mut)]
        let mut cuda = ort::execution_providers::CUDAExecutionProvider::default();
        #[cfg(feature = "cuda_tf32")]
        {
            cuda = cuda.with_tf32(true);
        }
        exp.push(cuda.build());
    }
    #[cfg(feature = "directml")]
    {
        exp.push(ort::execution_providers::DirectMLExecutionProvider::default().build());
    }
    #[cfg(feature = "coreml")]
    {
        exp.push(ort::execution_providers::CoreMLExecutionProvider::default().build());
    }
    exp.push(ort::execution_providers::CPUExecutionProvider::default().build());
    let builder = Session::builder()?
        .with_execution_providers(exp)?
        .with_parallel_execution(true)?;
    if global_pool() {
        return Ok(builder);
    }
    Ok(builder
        .with_intra_threads(num_cpus::get_physical())?
        .with_inter_threads(num_cpus::get_physical())?)
}

#[allow(clippy::too_many_arguments)]
pub fn synthesize(
    session: &mut Session,
    bert_ori: Array2<f32>,
    x_tst: Array1<i64>,
    mut spk_ids: Array1<i64>,
    tones: Array1<i64>,
    lang_ids: Array1<i64>,
    style_vector: Array1<f32>,
    sdp_ratio: f32,
    length_scale: f32,
    noise_scale: f32,
    noise_scale_w: f32,
) -> Result<Array3<f32>> {
    let bert_ori = bert_ori.insert_axis(Axis(0));
    let bert_ori = bert_ori.as_standard_layout();
    let bert = ort::value::TensorRef::from_array_view(&bert_ori)?;
    let mut x_tst_lengths = array![x_tst.shape()[0] as i64];
    let x_tst_lengths = ort::value::TensorRef::from_array_view(&mut x_tst_lengths)?;
    let mut x_tst = x_tst.insert_axis(Axis(0));
    let x_tst = ort::value::TensorRef::from_array_view(&mut x_tst)?;
    let mut lang_ids = lang_ids.insert_axis(Axis(0));
    let lang_ids = ort::value::TensorRef::from_array_view(&mut lang_ids)?;
    let mut tones = tones.insert_axis(Axis(0));
    let tones = ort::value::TensorRef::from_array_view(&mut tones)?;
    let mut style_vector = style_vector.insert_axis(Axis(0));
    let style_vector = ort::value::TensorRef::from_array_view(&mut style_vector)?;
    let sid = ort::value::TensorRef::from_array_view(&mut spk_ids)?;
    let sdp_ratio = vec![sdp_ratio];
    let sdp_ratio = ort::value::TensorRef::from_array_view((vec![1_i64], sdp_ratio.as_slice()))?;
    let length_scale = vec![length_scale];
    let length_scale =
        ort::value::TensorRef::from_array_view((vec![1_i64], length_scale.as_slice()))?;
    let noise_scale = vec![noise_scale];
    let noise_scale =
        ort::value::TensorRef::from_array_view((vec![1_i64], noise_scale.as_slice()))?;
    let noise_scale_w = vec![noise_scale_w];
    let noise_scale_w =
        ort::value::TensorRef::from_array_view((vec![1_i64], noise_scale_w.as_slice()))?;
    let outputs = session.run(ort::inputs! {
        "x_tst" =>  x_tst,
        "x_tst_lengths" => x_tst_lengths,
        "sid" => sid,
        "tones" => tones,
        "language" => lang_ids,
        "bert" => bert,
        "style_vec" => style_vector,
        "sdp_ratio" => sdp_ratio,
        "length_scale" => length_scale,
        "noise_scale" => noise_scale,
        "noise_scale_w" => noise_scale_w,
    })?;
    let audio_array = outputs["output"]
        .try_extract_array::<f32>()?
        .into_dimensionality::<Ix3>()?
        .to_owned();
    Ok(audio_array)
}
