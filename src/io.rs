//! Zero-allocation numeric reader.
//!
//! Reads the whole input into one buffer and splits on ASCII whitespace in
//! place, parsing each token with `fast_float2` (Lemire's algorithm). No
//! per-line `String`, which is what makes the parse beat NumPy's `loadtxt`.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use rsomics_common::{Result, RsomicsError};

/// Parse a whitespace-separated column of f64 values from a file (`-` = stdin).
pub fn read_values(path: &Path) -> Result<Vec<f64>> {
    let mut buf = Vec::new();
    if path.as_os_str() == "-" {
        std::io::stdin()
            .lock()
            .read_to_end(&mut buf)
            .map_err(RsomicsError::Io)?;
    } else {
        File::open(path)
            .map_err(RsomicsError::Io)?
            .read_to_end(&mut buf)
            .map_err(RsomicsError::Io)?;
    }
    parse_buffer(&buf)
}

/// Parse every ASCII-whitespace-delimited token of `buf` as f64.
pub fn parse_buffer(buf: &[u8]) -> Result<Vec<f64>> {
    let mut values = Vec::new();
    let mut start = None;
    for (i, &b) in buf.iter().enumerate() {
        if b.is_ascii_whitespace() {
            if let Some(s) = start.take() {
                push_token(&buf[s..i], &mut values)?;
            }
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        push_token(&buf[s..], &mut values)?;
    }
    if values.is_empty() {
        return Err(RsomicsError::InvalidInput("no values in input".into()));
    }
    Ok(values)
}

fn push_token(tok: &[u8], out: &mut Vec<f64>) -> Result<()> {
    let v: f64 = fast_float2::parse(tok).map_err(|_| {
        RsomicsError::InvalidInput(format!(
            "value '{}' is not a number",
            String::from_utf8_lossy(tok)
        ))
    })?;
    out.push(v);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_buffer;

    #[test]
    fn parses_newline_column() {
        assert_eq!(parse_buffer(b"1\n2.5\n-3\n").unwrap(), vec![1.0, 2.5, -3.0]);
    }

    #[test]
    fn parses_mixed_whitespace() {
        assert_eq!(
            parse_buffer(b"  1.0  2.0\t3.0\r\n4.0 ").unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn rejects_non_numeric() {
        assert!(parse_buffer(b"1\nfoo\n").is_err());
    }

    #[test]
    fn rejects_empty() {
        assert!(parse_buffer(b"  \n\t\n").is_err());
    }
}
