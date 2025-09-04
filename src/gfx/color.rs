use std::fmt;
use std::str::FromStr;

///////////////////////////////////////////////////////////////////////////
// Rgba8
///////////////////////////////////////////////////////////////////////////

/// RGBA color with 8-bit channels.
#[repr(C)]
#[derive(Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Debug, Default)]
pub struct Rgba8 {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Rgba8 {
    pub const TRANSPARENT: Self = Self {
        r: 0,
        g: 0,
        b: 0,
        a: 0,
    };
    pub const WHITE: Self = Self {
        r: 0xff,
        g: 0xff,
        b: 0xff,
        a: 0xff,
    };
    pub const BLACK: Self = Self {
        r: 0,
        g: 0,
        b: 0,
        a: 0xff,
    };
    pub const RED: Self = Self {
        r: 0xff,
        g: 0,
        b: 0,
        a: 0xff,
    };
    pub const GREEN: Self = Self {
        r: 0,
        g: 0xff,
        b: 0,
        a: 0xff,
    };
    pub const BLUE: Self = Self {
        r: 0,
        g: 0,
        b: 0xff,
        a: 0xff,
    };

    /// Create a new [`Rgba8`] color from individual channels.
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Invert the color.
    pub fn invert(self) -> Self {
        Self::new(0xff - self.r, 0xff - self.g, 0xff - self.b, self.a)
    }

    /// Return the color with a changed alpha.
    ///
    /// ```
    /// use rx::gfx::color::Rgba8;
    ///
    /// let c = Rgba8::WHITE;
    /// assert_eq!(c.alpha(0x88), Rgba8::new(c.r, c.g, c.b, 0x88))
    /// ```
    pub fn alpha(self, a: u8) -> Self {
        Self::new(self.r, self.g, self.b, a)
    }

    /// Given a byte slice, returns a slice of [`Rgba8`] values.
    pub fn align<'a, S: 'a, T: AsRef<[S]> + ?Sized>(bytes: &'a T) -> &'a [Rgba8] {
        let bytes = bytes.as_ref();
        let (head, body, tail) = unsafe { bytes.align_to::<Rgba8>() };

        if !(head.is_empty() && tail.is_empty()) {
            panic!("Rgba8::align: input is not a valid Rgba8 buffer");
        }
        body
    }
}

/// ```
/// use rx::gfx::color::Rgba8;
///
/// assert_eq!(format!("{}", Rgba8::new(0xff, 0x0, 0xa, 0xff)), "#ff000a");
/// ```
impl fmt::Display for Rgba8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{:02x}{:02x}{:02x}", self.r, self.g, self.b)?;
        if self.a != 0xff {
            write!(f, "{:02x}", self.a)?;
        }
        Ok(())
    }
}

/// ```
/// use rx::gfx::color::{Rgba8, Rgba};
///
/// assert_eq!(Rgba8::from(Rgba::RED), Rgba8::RED);
/// ```
impl From<Rgba> for Rgba8 {
    fn from(rgba: Rgba) -> Self {
        Self {
            r: (rgba.r * 255.0).round() as u8,
            g: (rgba.g * 255.0).round() as u8,
            b: (rgba.b * 255.0).round() as u8,
            a: (rgba.a * 255.0).round() as u8,
        }
    }
}

impl From<u32> for Rgba8 {
    fn from(rgba: u32) -> Self {
        unsafe { std::mem::transmute(rgba) }
    }
}

impl FromStr for Rgba8 {
    type Err = std::num::ParseIntError;

    /// Parse a color code of the form `#ffffff` into an
    /// instance of `Rgba8`. The alpha is always `0xff`.
    fn from_str(hex_code: &str) -> Result<Self, Self::Err> {
        let r: u8 = u8::from_str_radix(&hex_code[1..3], 16)?;
        let g: u8 = u8::from_str_radix(&hex_code[3..5], 16)?;
        let b: u8 = u8::from_str_radix(&hex_code[5..7], 16)?;
        let a: u8 = 0xff;

        Ok(Rgba8 { r, g, b, a })
    }
}

//////////////////////////////////////////////////////////////////////////////
// Rgb8
//////////////////////////////////////////////////////////////////////////////

/// An RGB 8-bit color. Used when the alpha value isn't used.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Rgb8 {
    r: u8,
    g: u8,
    b: u8,
}

impl From<Rgba8> for Rgb8 {
    fn from(rgba: Rgba8) -> Self {
        Self {
            r: rgba.r,
            g: rgba.g,
            b: rgba.b,
        }
    }
}

impl fmt::Display for Rgb8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{:02X}{:02X}{:02X}", self.r, self.g, self.b)
    }
}

//////////////////////////////////////////////////////////////////////////////
// Rgba
//////////////////////////////////////////////////////////////////////////////

/// A normalized RGBA color.
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Debug, Default)]
pub struct Rgba {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Rgba {
    pub const RED: Self = Rgba::new(1.0, 0.0, 0.0, 1.0);
    pub const GREEN: Self = Rgba::new(0.0, 1.0, 0.0, 1.0);
    pub const BLUE: Self = Rgba::new(0.0, 0.0, 1.0, 1.0);
    pub const WHITE: Self = Rgba::new(1.0, 1.0, 1.0, 1.0);
    pub const BLACK: Self = Rgba::new(0.0, 0.0, 0.0, 1.0);
    pub const TRANSPARENT: Self = Rgba::new(0.0, 0.0, 0.0, 0.0);

    /// Create a new `Rgba` color.
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// Invert the color.
    pub fn invert(self) -> Self {
        Self::new(1.0 - self.r, 1.0 - self.g, 1.0 - self.b, self.a)
    }
}

impl From<Rgba8> for Rgba {
    fn from(rgba8: Rgba8) -> Self {
        Self {
            r: (rgba8.r as f32 / 255.0),
            g: (rgba8.g as f32 / 255.0),
            b: (rgba8.b as f32 / 255.0),
            a: (rgba8.a as f32 / 255.0),
        }
    }
}

impl From<Hsv> for Rgba {
    fn from(hsv: Hsv) -> Self {
        let h = hsv.h;
        let s = hsv.s;
        let v = hsv.v;
        
        let c = v * s;
        let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
        let m = v - c;
        
        let (r, g, b) = match (h / 60.0) as i32 {
            0 => (c, x, 0.0),
            1 => (x, c, 0.0),
            2 => (0.0, c, x),
            3 => (0.0, x, c),
            4 => (x, 0.0, c),
            5 => (c, 0.0, x),
            _ => (0.0, 0.0, 0.0),
        };
        
        Rgba {
            r: r + m,
            g: g + m,
            b: b + m,
            a: 1.0,
        }
    }
}

/// A normalized HSV color.
#[derive(Copy, Clone, PartialEq, Debug, Default)]
pub struct Hsv {
    pub h: f32, // in degrees
    pub s: f32, // 0-1
    pub v: f32, // 0-1
}

impl From<Rgba> for Hsv {
    fn from(rgba: Rgba) -> Self {
        let r = rgba.r;
        let g = rgba.g;
        let b = rgba.b;
        
        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let delta = max - min;
        
        let mut h = 0.0;
        let s = if max > 0.0 { delta / max } else { 0.0 };
        let v = max;
        
        if delta != 0.0 {
            h = match max {
                x if x == r => 60.0 * (((g - b) / delta) % 6.0),
                x if x == g => 60.0 * (((b - r) / delta) + 2.0),
                x if x == b => 60.0 * (((r - g) / delta) + 4.0),
                _ => 0.0,
            };
            
            if h < 0.0 {
                h += 360.0;
            }
        }
        
        Hsv { h, s, v }
    }
}

pub fn generate_color_grid(
    hue_deg: f32, 
    w: u32, 
    h: u32, 
    gamma: f32, 
    gamma_v: f32,
    s_start: f32,
    s_end: f32,
    v_start: f32,
    v_end: f32
) -> impl Iterator<Item = Rgba8>
{
    // Normalize hue into [0, 360)
    let mut hue = hue_deg % 360.0;
    if hue < 0.0 { hue += 360.0; }

    let w = w.max(1);
    let h = h.max(1);
    let g  = if gamma   > 0.0 { gamma   } else { 1.0 };
    let gv = if gamma_v > 0.0 { gamma_v } else { 1.0 };

    // Precompute denominators to avoid div-by-zero.
    let w_denom = if w > 1 { (w - 1) as f32 } else { 1.0 };
    let h_denom = if h > 1 { (h - 1) as f32 } else { 1.0 };

    // Build once; return its iterator.
    let mut out = Vec::with_capacity((w as usize) * (h as usize));

    // Determine which dimension controls S vs V based on width vs height
    let (s_denom, v_denom) = if w > h {
        // Width > Height: width controls V, height controls S
        (h_denom, w_denom)
    } else {
        // Height >= Width: height controls V, width controls S
        (w_denom, h_denom)
    };

    for row in 0..h {
        for col in 0..w {
            // Calculate S and V based on position and dimension mapping
            let (s, v) = if w > h {
                // Width > Height: row controls S, col controls V
                let t_s = (row as f32) / s_denom;        // 0..1
                let t_v = (col as f32) / v_denom;        // 0..1
                
                let s = s_start + (s_end - s_start) * t_s.powf(g).clamp(0.0, 1.0);
                let v = v_start + (v_end - v_start) * (1.0 - t_v.powf(gv)).clamp(0.0, 1.0);
                (s, v)
            } else {
                // Height >= Width: col controls S, row controls V
                let t_s = (col as f32) / s_denom;        // 0..1
                let t_v = (row as f32) / v_denom;        // 0..1
                
                let s = s_start + (s_end - s_start) * t_s.powf(g).clamp(0.0, 1.0);
                let v = v_start + (v_end - v_start) * (1.0 - t_v.powf(gv)).clamp(0.0, 1.0);
                (s, v)
            };

            let hsv = Hsv { h: hue, s: s.clamp(0.0, 1.0), v: v.clamp(0.0, 1.0) };
            let rgba: Rgba = hsv.into();
            out.push(Rgba8::from(rgba));
        }
    }

    out.into_iter()
}