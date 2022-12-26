#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: (Vec<f32>, Vec<usize>, usize)| {
    let (vertices, hole_indices, dims) = data;
    for vertex in &vertices {
        if !vertex.is_finite() {
            return;
        }
    }
    earcutr::earcut(&vertices, &hole_indices, 2);
});
