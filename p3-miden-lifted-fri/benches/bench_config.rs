pub const DEFAULT_LOG_TRACE_HEIGHTS: &[usize] = &[16, 17, 18];

pub fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|val| val.parse::<usize>().ok())
        .unwrap_or(default)
}

pub fn env_log_trace_heights() -> Vec<usize> {
    match std::env::var("P3_LOG_TRACE_HEIGHTS") {
        Ok(raw) => raw
            .split(',')
            .filter_map(|val| val.trim().parse::<usize>().ok())
            .collect(),
        Err(_) => DEFAULT_LOG_TRACE_HEIGHTS.to_vec(),
    }
}
