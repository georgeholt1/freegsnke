use std::f64::consts::PI;

/// Complete elliptic integral of the first kind K(m) where m = k^2.
#[inline]
pub fn ellipk(m: f64) -> f64 {
    if m <= 0.0 {
        return PI / 2.0;
    }
    if m >= 1.0 {
        return f64::INFINITY;
    }
    let mut a = 1.0;
    let mut b = (1.0 - m).sqrt();
    for _ in 0..16 {
        let a_next = 0.5 * (a + b);
        let b_next = (a * b).sqrt();
        if (a - b).abs() <= 1e-15 * a {
            a = a_next;
            break;
        }
        a = a_next;
        b = b_next;
    }
    PI / (2.0 * a)
}

/// Complete elliptic integral of the second kind E(m) where m = k^2.
#[inline]
pub fn ellipe(m: f64) -> f64 {
    if m <= 0.0 {
        return PI / 2.0;
    }
    if m >= 1.0 {
        return 1.0;
    }
    let mut a = 1.0;
    let mut b = (1.0 - m).sqrt();
    let mut sum = 0.5 * m;
    let mut two_pow = 1.0;

    for _ in 0..16 {
        let c = 0.5 * (a - b);
        sum += two_pow * c * c;
        two_pow *= 2.0;
        let a_next = 0.5 * (a + b);
        let b_next = (a * b).sqrt();
        if (a - b).abs() <= 1e-15 * a {
            a = a_next;
            break;
        }
        a = a_next;
        b = b_next;
    }
    let k = PI / (2.0 * a);
    k * (1.0 - sum)
}

/// Computes both K(m) and E(m) simultaneously.
#[inline]
pub fn ellipke(m: f64) -> (f64, f64) {
    if m <= 0.0 {
        return (PI / 2.0, PI / 2.0);
    }
    if m >= 1.0 {
        return (f64::INFINITY, 1.0);
    }
    let mut a = 1.0;
    let mut b = (1.0 - m).sqrt();
    let mut sum = 0.5 * m;
    let mut two_pow = 1.0;

    for _ in 0..16 {
        let c = 0.5 * (a - b);
        sum += two_pow * c * c;
        two_pow *= 2.0;
        let a_next = 0.5 * (a + b);
        let b_next = (a * b).sqrt();
        if (a - b).abs() <= 1e-15 * a {
            a = a_next;
            break;
        }
        a = a_next;
        b = b_next;
    }
    let k = PI / (2.0 * a);
    let e = k * (1.0 - sum);
    (k, e)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_known_values() {
        let (k0, e0) = ellipke(0.0);
        assert!((k0 - PI / 2.0).abs() < 1e-14);
        assert!((e0 - PI / 2.0).abs() < 1e-14);

        let (k05, e05) = ellipke(0.5);
        assert!((k05 - 1.854074677301372).abs() < 1e-13);
        assert!((e05 - 1.3506438810476755).abs() < 1e-13);
    }
}
