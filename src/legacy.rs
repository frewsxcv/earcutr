use num_traits::float::Float;
use std::fmt::Display;

use super::*;

#[allow(dead_code)]
pub fn dump<T: Float + Display>(ll: &LinkedLists<T>) -> String {
    let mut s = format!("LL, #nodes: {}", ll.nodes.len());
    s.push_str(&format!(
        " #used: {}\n",
        //        ll.nodes.len() as i64 - ll.freelist.len() as i64
        ll.nodes.len() as i64
    ));
    s.push_str(&format!(
        " {:>3} {:>3} {:>4} {:>4} {:>8.3} {:>8.3} {:>4} {:>4} {:>2} {:>2} {:>2} {:>4}\n",
        "vi", "i", "p", "n", "x", "y", "pz", "nz", "st", "fr", "cyl", "z"
    ));
    for n in &ll.nodes {
        s.push_str(&format!(
            " {:>3} {:>3} {:>4} {:>4} {:>8.3} {:>8.3} {:>4} {:>4} {:>2} {:>2} {:>2} {:>4}\n",
            n.idx,
            n.vertices_index,
            pn(n.prev_linked_list_node_index),
            pn(n.next_linked_list_node_index),
            n.x,
            n.y,
            pn(n.prevz_idx),
            pn(n.nextz_idx),
            pb(n.is_steiner_point),
            false,
            //            pb(ll.freelist.contains(&n.idx)),
            0, //,ll.iter(n.idx..n.idx).count(),
            n.z,
        ));
    }
    s
}

#[allow(dead_code)]
pub fn cycle_dump<T: Float + Display>(ll: &LinkedLists<T>, p: LinkedListNodeIndex) -> String {
    let mut s = format!("cycle from {}, ", p);
    s.push_str(&format!(" len {}, idxs:", 0)); //cycle_len(&ll, p)));
    let mut i = p;
    let end = i;
    let mut count = 0;
    loop {
        count += 1;
        s.push_str(&format!("{} ", &ll.nodes[i].idx));
        s.push_str(&format!("(i:{}), ", &ll.nodes[i].vertices_index));
        i = ll.nodes[i].next_linked_list_node_index;
        if i == end {
            break s;
        }
        if count > ll.nodes.len() {
            s.push_str(" infinite loop");
            break s;
        }
    }
}

pub fn pn(a: usize) -> String {
    match a {
        0x777A91CC => String::from("NULL"),
        _ => a.to_string(),
    }
}

pub fn pb(a: bool) -> String {
    match a {
        true => String::from("x"),
        false => String::from(" "),
    }
}

// turn a polygon in a multi-dimensional array form (e.g. as in GeoJSON)
// into a form Earcut accepts
pub fn flatten<T: Float + Display>(data: &Vec<Vec<Vec<T>>>) -> (Vec<T>, Vec<usize>, usize) {
    (
        data.iter().flatten().flatten().cloned().collect::<Vec<T>>(), // flat data
        data.iter()
            .take(data.len() - 1)
            .scan(0, |holeidx, v| {
                *holeidx += v.len();
                Some(*holeidx)
            })
            .collect::<Vec<usize>>(), // hole indexes
        data[0][0].len(),                                             // dimensions
    )
}

// return a percentage difference between the polygon area and its
// triangulation area; used to verify correctness of triangulation
pub fn deviation<T: Float + Display, V: crate::Vertices<T>>(
    vertices: &V,
    hole_indices: &[usize],
    dims: usize,
    triangles: &[usize],
) -> T {
    if DIM != dims {
        return T::nan();
    }
    let mut indices = hole_indices.to_vec();
    indices.push(vertices.len() / DIM);
    let (ix, iy) = (indices.iter(), indices.iter().skip(1));
    let body_area = vertices.signed_area(0, indices[0] * DIM).abs();
    let polygon_area = ix.zip(iy).fold(body_area, |a, (ix, iy)| {
        a - vertices.signed_area(ix * DIM, iy * DIM).abs()
    });

    let i = triangles.iter().skip(0).step_by(3).map(|x| x * DIM);
    let j = triangles.iter().skip(1).step_by(3).map(|x| x * DIM);
    let k = triangles.iter().skip(2).step_by(3).map(|x| x * DIM);
    let triangles_area = i.zip(j).zip(k).fold(T::zero(), |ta, ((a, b), c)| {
        ta + ((vertices.vertex(a) - vertices.vertex(c))
            * (vertices.vertex(b + 1) - vertices.vertex(a + 1))
            - (vertices.vertex(a) - vertices.vertex(b))
                * (vertices.vertex(c + 1) - vertices.vertex(a + 1)))
        .abs()
    });

    match polygon_area.is_zero() && triangles_area.is_zero() {
        true => T::zero(),
        false => ((triangles_area - polygon_area) / polygon_area).abs(),
    }
}
