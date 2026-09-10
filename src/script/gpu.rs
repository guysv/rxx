//! Parsing for the script GPU descriptors. Invalid/unknown fields fail before
//! creating GPU objects, and parsing borrows values so descriptors are reusable.
use rune::{runtime::Object, Value};

fn fields(object: &Object, allowed: &[&str]) -> Result<(), String> {
    for (key, _) in object.iter() {
        if !allowed.contains(&key.as_str()) {
            return Err(format!("unknown option `{key}`"));
        }
    }
    Ok(())
}
fn string(object: &Object, key: &str, default: Option<&str>) -> Result<String, String> {
    match object.get(key) {
        Some(value) => value
            .borrow_string_ref()
            .map(|s| s.to_owned())
            .map_err(|_| format!("`{key}` must be a string")),
        None => default
            .map(str::to_owned)
            .ok_or_else(|| format!("missing required option `{key}`")),
    }
}
fn factor(name: &str) -> Result<wgpu::BlendFactor, String> {
    use wgpu::BlendFactor::*;
    Ok(match name {
        "zero" => Zero,
        "one" => One,
        "src" => Src,
        "one_minus_src" => OneMinusSrc,
        "src_alpha" => SrcAlpha,
        "one_minus_src_alpha" => OneMinusSrcAlpha,
        "dst" => Dst,
        "one_minus_dst" => OneMinusDst,
        "dst_alpha" => DstAlpha,
        "one_minus_dst_alpha" => OneMinusDstAlpha,
        "src_alpha_saturated" => SrcAlphaSaturated,
        _ => return Err(format!("unknown blend factor `{name}`")),
    })
}
fn component(value: &Value) -> Result<wgpu::BlendComponent, String> {
    let object = value
        .borrow_ref::<Object>()
        .map_err(|_| "blend component must be an object")?;
    fields(&object, &["src", "dst", "op"])?;
    let operation = match string(&object, "op", None)?.as_str() {
        "add" => wgpu::BlendOperation::Add,
        "subtract" => wgpu::BlendOperation::Subtract,
        "reverse_subtract" => wgpu::BlendOperation::ReverseSubtract,
        "min" => wgpu::BlendOperation::Min,
        "max" => wgpu::BlendOperation::Max,
        other => return Err(format!("unknown blend operation `{other}`")),
    };
    Ok(wgpu::BlendComponent {
        src_factor: factor(&string(&object, "src", None)?)?,
        dst_factor: factor(&string(&object, "dst", None)?)?,
        operation,
    })
}
fn blend(value: &Value) -> Result<wgpu::BlendState, String> {
    if let Ok(name) = value.borrow_string_ref() {
        return match &*name {
            "replace" => Ok(wgpu::BlendState::REPLACE),
            "alpha" => Ok(wgpu::BlendState::ALPHA_BLENDING),
            "premultiplied_alpha" => Ok(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
            _ => Err(format!("unknown blend preset `{}`", &*name)),
        };
    }
    let object = value
        .borrow_ref::<Object>()
        .map_err(|_| "`blend` must be a preset or object")?;
    fields(&object, &["color", "alpha"])?;
    Ok(wgpu::BlendState {
        color: component(object.get("color").ok_or("missing blend `color`")?)?,
        alpha: component(object.get("alpha").ok_or("missing blend `alpha`")?)?,
    })
}

pub(super) struct PipelineOptions {
    pub vertex: String,
    pub fragment: String,
    pub textures: i64,
    pub topology: wgpu::PrimitiveTopology,
    pub sprite: bool,
    pub blend: wgpu::BlendState,
    pub format: wgpu::TextureFormat,
}
impl PipelineOptions {
    pub fn parse(object: &Object) -> Result<Self, String> {
        fields(
            object,
            &[
                "vertex",
                "fragment",
                "textures",
                "topology",
                "vertex_layout",
                "blend",
                "format",
            ],
        )?;
        let textures = match object.get("textures") {
            Some(value) => rune::from_value::<i64>(value.clone())
                .map_err(|_| "`textures` must be an integer")?,
            None => 0,
        };
        if !(0..=super::Gfx::MAX_TEXTURE_GROUPS).contains(&textures) {
            return Err("`textures` must be in 0..=3".into());
        }
        let topology = match string(object, "topology", Some("triangle_list"))?.as_str() {
            "triangle_list" => wgpu::PrimitiveTopology::TriangleList,
            "triangle_strip" => wgpu::PrimitiveTopology::TriangleStrip,
            "point_list" => wgpu::PrimitiveTopology::PointList,
            "line_list" => wgpu::PrimitiveTopology::LineList,
            "line_strip" => wgpu::PrimitiveTopology::LineStrip,
            other => return Err(format!("unknown topology `{other}`")),
        };
        let sprite = match string(object, "vertex_layout", Some("sprite"))?.as_str() {
            "sprite" => true,
            "none" => false,
            other => return Err(format!("unknown vertex layout `{other}`")),
        };
        let format = match string(object, "format", Some("rgba8unorm-srgb"))?.as_str() {
            "rgba8unorm-srgb" => wgpu::TextureFormat::Rgba8UnormSrgb,
            "rgba8unorm" => wgpu::TextureFormat::Rgba8Unorm,
            other => return Err(format!("unsupported target format `{other}`")),
        };
        Ok(Self {
            vertex: string(object, "vertex", None)?,
            fragment: string(object, "fragment", None)?,
            textures,
            topology,
            sprite,
            format,
            blend: blend(
                object
                    .get("blend")
                    .ok_or("missing required option `blend`")?,
            )?,
        })
    }
}

pub(super) struct PassOptions {
    pub label: String,
    pub load: wgpu::LoadOp<wgpu::Color>,
}
impl PassOptions {
    pub fn parse(object: &Object) -> Result<Self, String> {
        fields(object, &["label", "load", "clear"])?;
        let load = match string(object, "load", None)?.as_str() {
            "load" => {
                if object.get("clear").is_some() {
                    return Err("`clear` requires load: \"clear\"".into());
                }
                wgpu::LoadOp::Load
            }
            "clear" => {
                let color = match object.get("clear") {
                    None => wgpu::Color::TRANSPARENT,
                    Some(value) => {
                        let vector = value
                            .borrow_ref::<rune::runtime::Vec>()
                            .map_err(|_| "`clear` must be four linear RGBA floats")?;
                        let values = vector
                            .iter()
                            .map(|v| rune::from_value::<f64>(v.clone()))
                            .collect::<Result<Vec<_>, _>>()
                            .map_err(|_| "`clear` must be four linear RGBA floats")?;
                        if values.len() != 4
                            || values
                                .iter()
                                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
                        {
                            return Err("`clear` must be four finite floats in 0..=1".into());
                        }
                        wgpu::Color {
                            r: values[0],
                            g: values[1],
                            b: values[2],
                            a: values[3],
                        }
                    }
                };
                wgpu::LoadOp::Clear(color)
            }
            other => return Err(format!("unknown load op `{other}`")),
        };
        Ok(Self {
            label: string(object, "label", Some("script_pass"))?,
            load,
        })
    }
}
