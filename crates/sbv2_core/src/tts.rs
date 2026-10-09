use crate::error::{Error, Result};
use crate::{jtalk, model, style, tokenizer, tts_util};
#[cfg(feature = "aivmx")]
use base64::prelude::{Engine as _, BASE64_STANDARD};
#[cfg(feature = "aivmx")]
use ndarray::ShapeBuilder;
use ndarray::{concatenate, Array1, Array2, Array3, Axis};
use ort::session::Session;
#[cfg(feature = "aivmx")]
use std::io::Cursor;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

#[derive(PartialEq, Eq, Clone)]
pub struct TTSIdent(String);

impl std::fmt::Display for TTSIdent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)?;
        Ok(())
    }
}

impl<S> From<S> for TTSIdent
where
    S: AsRef<str>,
{
    fn from(value: S) -> Self {
        TTSIdent(value.as_ref().to_string())
    }
}

pub struct TTSModel {
    vits2: Option<Session>,
    style_vectors: Array2<f32>,
    ident: TTSIdent,
    source: Option<ModelSource>,
}

/// Where an unloaded model's vits2 onnx is rebuilt from
enum ModelSource {
    Bytes(Vec<u8>),
    OnnxFile(PathBuf),
    Sbv2File(PathBuf),
}

impl ModelSource {
    fn read(&self) -> Result<Vec<u8>> {
        match self {
            ModelSource::Bytes(b) => Ok(b.clone()),
            ModelSource::OnnxFile(p) => Ok(std::fs::read(p)?),
            ModelSource::Sbv2File(p) => Ok(crate::sbv2file::parse_sbv2file(std::fs::read(p)?)?.1),
        }
    }
}

/// High-level Style-Bert-VITS2's API
pub struct TTSModelHolder {
    tokenizer: Tokenizer,
    bert: Session,
    models: Vec<TTSModel>,
    pub jtalk: jtalk::JTalk,
    max_loaded_models: Option<usize>,
    optimized_dir: Option<PathBuf>,
}

impl TTSModelHolder {
    /// Initialize a new TTSModelHolder
    ///
    /// # Examples
    ///
    /// ```rs
    /// let mut tts_holder = TTSModelHolder::new(std::fs::read("deberta.onnx")?, std::fs::read("tokenizer.json")?, None)?;
    /// ```
    pub fn new<P: AsRef<[u8]>>(
        bert_model_bytes: P,
        tokenizer_bytes: P,
        max_loaded_models: Option<usize>,
    ) -> Result<Self> {
        let bert = model::load_model(bert_model_bytes, true)?;
        let jtalk = jtalk::JTalk::new()?;
        let tokenizer = tokenizer::get_tokenizer(tokenizer_bytes)?;
        Ok(TTSModelHolder {
            bert,
            models: vec![],
            jtalk,
            tokenizer,
            max_loaded_models,
            optimized_dir: None,
        })
    }

    /// Write each voice's optimized graph to `dir` and reload evicted voices from it,
    /// skipping graph optimization. The graph is specific to this machine.
    pub fn set_optimized_dir<D: Into<PathBuf>>(&mut self, dir: D) {
        self.optimized_dir = Some(dir.into());
    }

    fn optimized_path(&self, ident: &TTSIdent) -> Option<PathBuf> {
        self.optimized_dir
            .as_ref()
            .map(|d| d.join(format!("{ident}.onnx")))
    }

    fn build_vits2(&self, ident: &TTSIdent, vits2_bytes: &[u8]) -> Result<Session> {
        match self.optimized_path(ident) {
            Some(p) if p.exists() => model::load_optimized_model(&p, false),
            Some(p) => model::load_model_saving_optimized(vits2_bytes, false, &p),
            None => model::load_model(vits2_bytes, false),
        }
    }

    /// Return a list of model names
    pub fn models(&self) -> Vec<String> {
        self.models.iter().map(|m| m.ident.to_string()).collect()
    }

    #[cfg(feature = "aivmx")]
    pub fn load_aivmx<I: Into<TTSIdent>, P: AsRef<[u8]>>(
        &mut self,
        ident: I,
        aivmx_bytes: P,
    ) -> Result<()> {
        self.load_aivmx_with(ident, &aivmx_bytes, || {
            ModelSource::Bytes(aivmx_bytes.as_ref().to_vec())
        })
    }

    /// Load a .aivmx file; an evicted session is rebuilt by reading the file again
    /// instead of keeping its bytes in memory.
    #[cfg(feature = "aivmx")]
    pub fn load_aivmx_path<I: Into<TTSIdent>, F: AsRef<Path>>(
        &mut self,
        ident: I,
        path: F,
    ) -> Result<()> {
        let path = path.as_ref();
        let aivmx_bytes = std::fs::read(path)?;
        self.load_aivmx_with(ident, &aivmx_bytes, || {
            ModelSource::OnnxFile(path.to_path_buf())
        })
    }

    #[cfg(feature = "aivmx")]
    fn load_aivmx_with<I: Into<TTSIdent>, P: AsRef<[u8]>>(
        &mut self,
        ident: I,
        aivmx_bytes: P,
        source: impl FnOnce() -> ModelSource,
    ) -> Result<()> {
        let ident = ident.into();
        if self.find_model(ident.clone()).is_err() {
            let mut load = true;
            if let Some(max) = self.max_loaded_models {
                if self.models.iter().filter(|x| x.vits2.is_some()).count() >= max {
                    load = false;
                }
            }
            let model = self.build_vits2(&ident, aivmx_bytes.as_ref())?;
            let metadata = model.metadata()?;
            if let Some(aivm_style_vectors) = metadata.custom("aivm_style_vectors") {
                let aivm_style_vectors = BASE64_STANDARD.decode(aivm_style_vectors)?;
                let style_vectors = Cursor::new(&aivm_style_vectors);
                let reader = npyz::NpyFile::new(style_vectors)?;
                let style_vectors = {
                    let shape = reader.shape().to_vec();
                    let order = reader.order();
                    let data = reader.into_vec::<f32>()?;
                    let shape = match shape[..] {
                        [i1, i2] => [i1 as usize, i2 as usize],
                        _ => panic!("expected 2D array"),
                    };
                    let true_shape = shape.set_f(order == npyz::Order::Fortran);
                    ndarray::Array2::from_shape_vec(true_shape, data)?
                };
                drop(metadata);
                self.models.push(TTSModel {
                    vits2: if load { Some(model) } else { None },
                    source: if self.max_loaded_models.is_some() {
                        Some(source())
                    } else {
                        None
                    },
                    ident,
                    style_vectors,
                })
            }
        }
        Ok(())
    }

    /// Load a .sbv2 file binary
    ///
    /// # Examples
    ///
    /// ```rs
    /// tts_holder.load_sbv2file("tsukuyomi", std::fs::read("tsukuyomi.sbv2")?)?;
    /// ```
    pub fn load_sbv2file<I: Into<TTSIdent>, P: AsRef<[u8]>>(
        &mut self,
        ident: I,
        sbv2_bytes: P,
    ) -> Result<()> {
        let (style_vectors, vits2) = crate::sbv2file::parse_sbv2file(sbv2_bytes)?;
        self.load(ident, style_vectors, vits2)?;
        Ok(())
    }

    /// Load a .sbv2 file; an evicted session is rebuilt by reading the file again
    /// instead of keeping its bytes in memory.
    pub fn load_sbv2file_path<I: Into<TTSIdent>, F: AsRef<Path>>(
        &mut self,
        ident: I,
        path: F,
    ) -> Result<()> {
        let path = path.as_ref();
        let (style_vectors, vits2) =
            crate::sbv2file::parse_sbv2file(std::fs::read(path)?)?;
        self.load_with(ident, style_vectors, vits2, || {
            ModelSource::Sbv2File(path.to_path_buf())
        })
    }

    /// Load a style vector and onnx model binary
    ///
    /// # Examples
    ///
    /// ```rs
    /// tts_holder.load("tsukuyomi", std::fs::read("style_vectors.json")?, std::fs::read("model.onnx")?)?;
    /// ```
    pub fn load<I: Into<TTSIdent>, P: AsRef<[u8]>>(
        &mut self,
        ident: I,
        style_vectors_bytes: P,
        vits2_bytes: P,
    ) -> Result<()> {
        self.load_with(ident, style_vectors_bytes, &vits2_bytes, || {
            ModelSource::Bytes(vits2_bytes.as_ref().to_vec())
        })
    }

    fn load_with<I: Into<TTSIdent>, P: AsRef<[u8]>, Q: AsRef<[u8]>>(
        &mut self,
        ident: I,
        style_vectors_bytes: P,
        vits2_bytes: Q,
        source: impl FnOnce() -> ModelSource,
    ) -> Result<()> {
        let ident = ident.into();
        if self.find_model(ident.clone()).is_err() {
            let mut load = true;
            if let Some(max) = self.max_loaded_models {
                if self.models.iter().filter(|x| x.vits2.is_some()).count() >= max {
                    load = false;
                }
            }
            self.models.push(TTSModel {
                vits2: if load {
                    Some(self.build_vits2(&ident, vits2_bytes.as_ref())?)
                } else {
                    None
                },
                style_vectors: style::load_style(style_vectors_bytes)?,
                ident,
                source: if self.max_loaded_models.is_some() {
                    Some(source())
                } else {
                    None
                },
            })
        }
        Ok(())
    }

    /// Unload a model
    pub fn unload<I: Into<TTSIdent>>(&mut self, ident: I) -> bool {
        let ident = ident.into();
        if let Some((i, _)) = self
            .models
            .iter()
            .enumerate()
            .find(|(_, m)| m.ident == ident)
        {
            self.models.remove(i);
            true
        } else {
            false
        }
    }

    /// Parse text and return the input for synthesize
    ///
    /// # Note
    /// This function is for low-level usage, use `easy_synthesize` for high-level usage.
    #[allow(clippy::type_complexity)]
    pub fn parse_text(
        &mut self,
        text: &str,
    ) -> Result<(Array2<f32>, Array1<i64>, Array1<i64>, Array1<i64>)> {
        crate::tts_util::parse_text_blocking(
            text,
            None,
            &self.jtalk,
            &self.tokenizer,
            |token_ids, attention_masks| {
                crate::bert::predict(&mut self.bert, token_ids, attention_masks)
            },
        )
    }

    #[allow(clippy::type_complexity)]
    pub fn parse_text_neo(
        &mut self,
        text: String,
        given_tones: Option<Vec<i32>>,
    ) -> Result<(Array2<f32>, Array1<i64>, Array1<i64>, Array1<i64>)> {
        crate::tts_util::parse_text_blocking(
            &text,
            given_tones,
            &self.jtalk,
            &self.tokenizer,
            |token_ids, attention_masks| {
                crate::bert::predict(&mut self.bert, token_ids, attention_masks)
            },
        )
    }

    fn find_model<I: Into<TTSIdent>>(&mut self, ident: I) -> Result<&mut TTSModel> {
        let ident = ident.into();
        self.models
            .iter_mut()
            .find(|m| m.ident == ident)
            .ok_or(Error::ModelNotFoundError(ident.to_string()))
    }
    fn find_and_load_model<I: Into<TTSIdent>>(&mut self, ident: I) -> Result<bool> {
        let ident = ident.into();
        // Locate target model entry
        let target_index = self
            .models
            .iter()
            .position(|m| m.ident == ident)
            .ok_or(Error::ModelNotFoundError(ident.to_string()))?;

        // Already loaded
        if self.models[target_index].vits2.is_some() {
            return Ok(true);
        }

        // Get the optimized graph, or bytes to build a Session
        let input = match self.optimized_path(&ident).filter(|p| p.exists()) {
            Some(p) => Ok(p),
            None => Err(self.models[target_index]
                .source
                .as_ref()
                .ok_or(Error::ModelNotFoundError(ident.to_string()))?
                .read()?),
        };

        // Enforce max loaded models by evicting a different loaded model's session, not removing the entry
        if let Some(max) = self.max_loaded_models {
            let loaded_count = self.models.iter().filter(|m| m.vits2.is_some()).count();
            if loaded_count >= max {
                if let Some(evict_index) = self
                    .models
                    .iter()
                    .position(|m| m.vits2.is_some() && m.ident != ident)
                {
                    // Drop only the session to free memory; keep bytes/style for future reload
                    self.models[evict_index].vits2 = None;
                }
            }
        }

        // Build and set session in-place for the target model
        let s = match input {
            Ok(p) => model::load_optimized_model(&p, false)?,
            Err(bytes) => self.build_vits2(&ident, &bytes)?,
        };
        self.models[target_index].vits2 = Some(s);
        Ok(true)
    }

    /// Get style vector by style id and weight
    ///
    /// # Note
    /// This function is for low-level usage, use `easy_synthesize` for high-level usage.
    pub fn get_style_vector<I: Into<TTSIdent>>(
        &mut self,
        ident: I,
        style_id: i32,
        weight: f32,
    ) -> Result<Array1<f32>> {
        style::get_style_vector(&self.find_model(ident)?.style_vectors, style_id, weight)
    }

    /// Synthesize text to audio
    ///
    /// # Examples
    ///
    /// ```rs
    /// let audio = tts_holder.easy_synthesize("tsukuyomi", "こんにちは", 0, SynthesizeOptions::default())?;
    /// ```
    pub fn easy_synthesize<I: Into<TTSIdent> + Copy>(
        &mut self,
        ident: I,
        text: &str,
        style_id: i32,
        speaker_id: i64,
        options: SynthesizeOptions,
    ) -> Result<Vec<u8>> {
        let result = self.easy_synthesize_loaded(ident, text, style_id, speaker_id, options);
        self.release_unretained();
        result
    }

    /// With `max_loaded_models` of 0, drop the sessions built for a call once it returns.
    fn release_unretained(&mut self) {
        if self.max_loaded_models == Some(0) {
            for m in &mut self.models {
                m.vits2 = None;
            }
        }
    }

    fn easy_synthesize_loaded<I: Into<TTSIdent> + Copy>(
        &mut self,
        ident: I,
        text: &str,
        style_id: i32,
        speaker_id: i64,
        options: SynthesizeOptions,
    ) -> Result<Vec<u8>> {
        self.find_and_load_model(ident)?;
        let style_vector = self.get_style_vector(ident, style_id, options.style_weight)?;
        let audio_array = if options.split_sentences {
            let texts: Vec<&str> = text.split('\n').collect();
            let mut audios = vec![];
            for (i, t) in texts.iter().enumerate() {
                if t.is_empty() {
                    continue;
                }
                let audio = self
                    .synthesize_segment_with_fallback(ident, t, &style_vector, speaker_id, &options)?;
                audios.push(audio);
                if i != texts.len() - 1 {
                    audios.push(Array3::zeros((1, 1, 22050)));
                }
            }
            concatenate(
                Axis(2),
                &audios.iter().map(|x| x.view()).collect::<Vec<_>>(),
            )?
        } else {
            self.synthesize_segment_with_fallback(ident, text, &style_vector, speaker_id, &options)?
        };
        tts_util::array_to_vec(audio_array)
    }

    /// Synthesize a single text into an audio array (no fallback / no `\n` splitting).
    fn synthesize_one<I: Into<TTSIdent> + Copy>(
        &mut self,
        ident: I,
        text: &str,
        style_vector: &Array1<f32>,
        speaker_id: i64,
        options: &SynthesizeOptions,
    ) -> Result<Array3<f32>> {
        let (bert_ori, phones, tones, lang_ids) = self.parse_text(text)?;
        let vits2 = self
            .find_model(ident)?
            .vits2
            .as_mut()
            .ok_or(Error::ModelNotFoundError(ident.into().to_string()))?;
        model::synthesize(
            vits2,
            bert_ori.to_owned(),
            phones,
            Array1::from_vec(vec![speaker_id]),
            tones,
            lang_ids,
            style_vector.clone(),
            options.sdp_ratio,
            options.length_scale,
            0.677,
            0.8,
        )
    }

    /// Synthesize `text`, but if the frontend (jpreprocess) or the model fails on it
    /// (e.g. jpreprocess `PartOfSpeechParseError` on words like `スペシャルゲスト`), split the
    /// text roughly in half at a char boundary and recurse on each half, concatenating the
    /// recovered audio with a small gap. Single unparseable units fall back to a short silence.
    ///
    /// This guarantees the request never fails on a single un-processable word (the whole
    /// upstream design flaw: it errors instead of degrading). Quality of the split part degrades
    /// (small gaps / choppier reading) but audio is always produced. Smooth reading of such words
    /// is a separate concern (fix jpreprocess to assign a default POS).
    fn synthesize_segment_with_fallback<I: Into<TTSIdent> + Copy>(
        &mut self,
        ident: I,
        text: &str,
        style_vector: &Array1<f32>,
        speaker_id: i64,
        options: &SynthesizeOptions,
    ) -> Result<Array3<f32>> {
        // Common path: try the whole segment first (best quality, single synthesis).
        if let Ok(audio) = self.synthesize_one(ident, text, style_vector, speaker_id, options) {
            return Ok(audio);
        }
        let chars: Vec<char> = text.chars().collect();
        // Base case: cannot split further -> emit a short silence instead of failing.
        if chars.len() <= 1 {
            return Ok(Array3::zeros((1, 1, 2205)));
        }
        // Split in half at a char boundary and recover each side recursively.
        let mid = chars.len() / 2;
        let left: String = chars[..mid].iter().collect();
        let right: String = chars[mid..].iter().collect();
        let left_audio =
            self.synthesize_segment_with_fallback(ident, &left, style_vector, speaker_id, options)?;
        let right_audio =
            self.synthesize_segment_with_fallback(ident, &right, style_vector, speaker_id, options)?;
        let gap = Array3::<f32>::zeros((1, 1, 2205));
        Ok(concatenate(
            Axis(2),
            &[left_audio.view(), gap.view(), right_audio.view()],
        )?)
    }

    pub fn easy_synthesize_neo<I: Into<TTSIdent> + Copy>(
        &mut self,
        ident: I,
        text: &str,
        given_tones: Option<Vec<i32>>,
        style_id: i32,
        speaker_id: i64,
        options: SynthesizeOptions,
    ) -> Result<Vec<u8>> {
        let result =
            self.easy_synthesize_neo_loaded(ident, text, given_tones, style_id, speaker_id, options);
        self.release_unretained();
        result
    }

    fn easy_synthesize_neo_loaded<I: Into<TTSIdent> + Copy>(
        &mut self,
        ident: I,
        text: &str,
        given_tones: Option<Vec<i32>>,
        style_id: i32,
        speaker_id: i64,
        options: SynthesizeOptions,
    ) -> Result<Vec<u8>> {
        self.find_and_load_model(ident)?;
        let style_vector = self.get_style_vector(ident, style_id, options.style_weight)?;
        let audio_array = if options.split_sentences {
            let texts: Vec<&str> = text.split('\n').collect();
            let mut audios = vec![];
            for (i, t) in texts.iter().enumerate() {
                if t.is_empty() {
                    continue;
                }
                let (bert_ori, phones, tones, lang_ids) =
                    self.parse_text_neo(t.to_string(), given_tones.clone())?;

                let vits2 = self
                    .find_model(ident)?
                    .vits2
                    .as_mut()
                    .ok_or(Error::ModelNotFoundError(ident.into().to_string()))?;
                let audio = model::synthesize(
                    vits2,
                    bert_ori.to_owned(),
                    phones,
                    Array1::from_vec(vec![speaker_id]),
                    tones,
                    lang_ids,
                    style_vector.clone(),
                    options.sdp_ratio,
                    options.length_scale,
                    0.677,
                    0.8,
                )?;
                audios.push(audio.clone());
                if i != texts.len() - 1 {
                    audios.push(Array3::zeros((1, 1, 22050)));
                }
            }
            concatenate(
                Axis(2),
                &audios.iter().map(|x| x.view()).collect::<Vec<_>>(),
            )?
        } else {
            let (bert_ori, phones, tones, lang_ids) = self.parse_text(text)?;

            let vits2 = self
                .find_model(ident)?
                .vits2
                .as_mut()
                .ok_or(Error::ModelNotFoundError(ident.into().to_string()))?;
            model::synthesize(
                vits2,
                bert_ori.to_owned(),
                phones,
                Array1::from_vec(vec![speaker_id]),
                tones,
                lang_ids,
                style_vector,
                options.sdp_ratio,
                options.length_scale,
                0.677,
                0.8,
            )?
        };
        tts_util::array_to_vec(audio_array)
    }
}

/// Synthesize options
///
/// # Fields
/// - `sdp_ratio`: SDP ratio
/// - `length_scale`: Length scale
/// - `style_weight`: Style weight
/// - `split_sentences`: Split sentences
pub struct SynthesizeOptions {
    pub sdp_ratio: f32,
    pub length_scale: f32,
    pub style_weight: f32,
    pub split_sentences: bool,
}

impl Default for SynthesizeOptions {
    fn default() -> Self {
        SynthesizeOptions {
            sdp_ratio: 0.0,
            length_scale: 1.0,
            style_weight: 1.0,
            split_sentences: true,
        }
    }
}
