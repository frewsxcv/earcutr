use super::*;

//static DEBUG: usize = 4;
static DEBUG: usize = 0; // dlogs get optimized away at 0

macro_rules! next {
    ($ll:expr,$idx:expr) => {
        $ll.nodes[$ll.nodes[$idx].next_linked_list_node_index]
    };
}
macro_rules! prev {
    ($ll:expr,$idx:expr) => {
        $ll.nodes[$ll.nodes[$idx].prev_linked_list_node_index]
    };
}

macro_rules! dlog {
	($loglevel:expr, $($s:expr),*) => (
		if DEBUG>=$loglevel { print!("{}:",$loglevel); println!($($s),+); }
	)
}

fn cycles_report<T: num_traits::float::Float + std::fmt::Display>(ll: &LinkedLists<T>) -> String {
    if ll.nodes.len() == 1 {
        return "[]".to_string();
    }
    let mut markv: Vec<usize> = Vec::new();
    markv.resize(ll.nodes.len(), NULL);
    let mut cycler;
    for i in 0..markv.len() {
        //            if ll.freelist.contains(&i) {
        if true {
            markv[i] = NULL;
        } else if markv[i] == NULL {
            cycler = i;
            let mut p = i;
            let end = ll.nodes[p].prev_linked_list_node_index;
            markv[p] = cycler;
            let mut count = 0;
            loop {
                p = ll.nodes[p].next_linked_list_node_index;
                markv[p] = cycler;
                count += 1;
                if p == end || count > ll.nodes.len() {
                    break;
                }
            } // loop
        } // if markvi == 0
    } //for markv
    format!("cycles report:\n{:?}", markv)
}

#[allow(dead_code)]
fn dump_cycle<T: num_traits::float::Float + std::fmt::Display>(
    ll: &LinkedLists<T>,
    start: usize,
) -> String {
    let mut s = format!("LL, #nodes: {}", ll.nodes.len());
    //        s.push_str(&format!(" #used: {}\n", ll.nodes.len() - ll.freelist.len()));
    s.push_str(&format!(" #used: {}\n", ll.nodes.len()));
    s.push_str(&format!(
        " {:>3} {:>3} {:>3} {:>4} {:>4} {:>8.3} {:>8.3} {:>4} {:>4} {:>2} {:>2} {:>2}\n",
        "#", "vi", "i", "p", "n", "x", "y", "pz", "nz", "st", "fr", "cyl"
    ));
    let mut startidx: usize = 0;
    for n in &ll.nodes {
        if n.vertices_index == start {
            startidx = n.idx;
        };
    }
    let endidx = startidx;
    let mut idx = startidx;
    let mut count = 0;
    let mut state; // = 0i32;
    loop {
        let n = &ll.nodes[idx].clone();
        state = 0; //horsh( state, n.i  as i32);
        s.push_str(&format!(
            " {:>3} {:>3} {:>3} {:>4} {:>4} {:>8.3} {:>8.3} {:>4} {:>4} {:>2} {:>2} {:>2}\n",
            count,
            n.idx,
            n.vertices_index,
            prev!(ll, n.idx).vertices_index,
            next!(ll, n.idx).vertices_index,
            n.x,
            n.y,
            crate::legacy::pn(n.prevz_idx),
            crate::legacy::pn(n.nextz_idx),
            crate::legacy::pb(n.is_steiner_point),
            //                pb(ll.freelist.contains(&n.idx)),
            false,
            cycle_len(ll, n.idx),
        ));
        idx = next!(ll, idx).idx;
        count += 1;
        if idx == endidx || count > ll.nodes.len() {
            break;
        }
    }
    s.push_str(&format!("dump end, horshcount:{} horsh:{}", count, state));
    s
}

fn cycle_len<T: num_traits::float::Float + std::fmt::Display>(
    ll: &LinkedLists<T>,
    p: LinkedListNodeIndex,
) -> usize {
    if p >= ll.nodes.len() {
        return 0;
    }
    let end = ll.nodes[p].prev_linked_list_node_index;
    let mut i = p;
    let mut count = 1;
    loop {
        i = ll.nodes[i].next_linked_list_node_index;
        count += 1;
        if i == end {
            break count;
        }
        if count > ll.nodes.len() {
            break count;
        }
    }
}

// https://www.cs.hmc.edu/~geoff/classes/hmc.cs070.200101/homework10/hashfuncs.$
// https://stackoverflow.com/questions/1908492/unsigned-integer-in-javascript
#[allow(dead_code)]
fn horsh(mut h: u32, n: u32) -> u32 {
    let highorder = h & 0xf8000000; // extract high-order 5 bits from h
                                    // 0xf8000000 is the hexadecimal representat$
                                    //   for the 32-bit number with the first fi$
                                    //   bits = 1 and the other bits = 0
    h <<= 5; // shift h left by 5 bits
    h ^= highorder >> 27; // move the highorder 5 bits to the low-ord$
                          //   end and XOR into h
    h ^= n; // XOR h and ki
    h
}

// find the node with 'i' of starti, horsh it
#[allow(dead_code)]
fn horsh_ll<T: num_traits::float::Float + std::fmt::Display>(
    ll: &LinkedLists<T>,
    starti: VerticesIndex,
) -> String {
    let mut s = "LL horsh: ".to_string();
    let mut startidx: usize = 0;
    for n in &ll.nodes {
        if n.vertices_index == starti {
            startidx = n.idx;
        };
    }
    let endidx = startidx;
    let mut idx = startidx;
    let mut count = 0;
    let mut state = 0u32;
    loop {
        let n = &ll.nodes[idx].clone();
        state = horsh(state, n.vertices_index as u32);
        idx = next!(ll, idx).idx;
        count += 1;
        if idx == endidx || count > ll.nodes.len() {
            break;
        }
    }
    s.push_str(&format!(" count:{} horsh: {}", count, state));
    s
}

#[test]
fn test_empty() {
    let result = earcut::<f32>(&[], &[], 2);
    assert_eq!(result, []);
}

#[test]
fn test_linked_list() {
    let vertices = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
    let (mut ll, _) = linked_list(&vertices, 0, vertices.len(), true);
    assert!(ll.nodes.len() == 5);
    assert!(ll.nodes[1].idx == 1);
    assert!(ll.nodes[1].vertices_index == 6 / DIM);
    assert!(ll.nodes[1].vertices_index == 3);
    assert!(ll.nodes[1].x == 1.0);
    assert!(ll.nodes[1].y == 0.0);
    assert!(
        ll.nodes[1].next_linked_list_node_index == 2
            && ll.nodes[1].prev_linked_list_node_index == 4
    );
    assert!(
        ll.nodes[4].next_linked_list_node_index == 1
            && ll.nodes[4].prev_linked_list_node_index == 3
    );
    ll.remove_node(2);
}

#[test]
fn test_iter_pairs() {
    let vertices = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    let (ll, _) = linked_list(&vertices, 0, vertices.len(), true);
    let mut v: Vec<LinkedListNode<f32>> = Vec::new();
    //        ll.iter(1..2)
    //.zip(ll.iter(2..3))
    ll.iter_pairs(1..2).for_each(|(p, n)| {
        v.push(*p);
        v.push(*n);
    });
    println!("{:?}", v);
    //		assert!(false);
}

#[test]
fn test_point_in_triangle() {
    let vertices = vec![0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 1.0, 0.1];
    let (ll, _) = linked_list(&vertices, 0, vertices.len(), true);
    assert!(point_in_triangle(
        ll.nodes[1],
        ll.nodes[2],
        ll.nodes[3],
        ll.nodes[4]
    ));
}

#[test]
fn test_signed_area() {
    let vertices1 = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
    let vertices2 = vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0];
    let a1 = signed_area(&vertices1, 0, 4);
    let a2 = signed_area(&vertices2, 0, 4);
    assert!(a1 == -a2);
}

#[test]
fn test_deviation() {
    let vertices1 = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
    let tris = vec![0, 1, 2, 2, 3, 0];
    let hi: Vec<usize> = Vec::new();
    assert!(deviation(&vertices1, &hi, DIM, &tris) == 0.0);
}

#[test]
fn test_split_bridge_polygon() {
    let mut body = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
    let hole = vec![0.1, 0.1, 0.1, 0.2, 0.2, 0.2];
    body.extend(hole);
    let (mut ll, _) = linked_list(&body, 0, body.len(), true);
    assert!(cycle_len(&ll, 1) == body.len() / DIM);
    let (left, right) = (1, 5);
    let np = split_bridge_polygon(&mut ll, left, right);
    assert!(cycle_len(&ll, left) == 4);
    assert!(cycle_len(&ll, np) == 5);
    // contrary to name, this should join the two cycles back together.
    let np2 = split_bridge_polygon(&mut ll, left, np);
    assert!(cycle_len(&ll, np2) == 11);
    assert!(cycle_len(&ll, left) == 11);
}

#[test]
fn test_equals() {
    let body = vec![0.0, 1.0, 0.0, 1.0];
    let (ll, _) = linked_list(&body, 0, body.len(), true);
    assert!(ll.nodes[1].xy_eq(ll.nodes[2]));

    let body = vec![2.0, 1.0, 0.0, 1.0];
    let (ll, _) = linked_list(&body, 0, body.len(), true);
    assert!(!ll.nodes[1].xy_eq(ll.nodes[2]));
}

#[test]
fn test_area() {
    let body = vec![4.0, 0.0, 4.0, 3.0, 0.0, 0.0]; // counterclockwise
    let (ll, _) = linked_list(&body, 0, body.len(), true);
    assert!(NodeTriangle(ll.nodes[1], ll.nodes[2], ll.nodes[3]).area() == -12.0);
    let body2 = vec![4.0, 0.0, 0.0, 0.0, 4.0, 3.0]; // clockwise
    let (ll2, _) = linked_list(&body2, 0, body2.len(), true);
    // creation apparently modifies all winding to ccw
    assert!(NodeTriangle(ll2.nodes[1], ll2.nodes[2], ll2.nodes[3]).area() == -12.0);
}

#[test]
fn test_is_ear() {
    let m = vec![0.0, 0.0, 0.5, 0.0, 1.0, 0.0];
    let (ll, _) = linked_list(&m, 0, m.len(), true);
    assert!(!NodeIndexTriangle(1, 2, 3).is_ear(&ll));
    assert!(!NodeIndexTriangle(2, 3, 1).is_ear(&ll));
    assert!(!NodeIndexTriangle(3, 1, 2).is_ear(&ll));

    let m = vec![0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.5, 0.4];
    let (ll, _) = linked_list(&m, 0, m.len(), true);
    assert!(!NodeIndexTriangle(4, 1, 2).is_ear(&ll));
    assert!(NodeIndexTriangle(1, 2, 3).is_ear(&ll));
    assert!(!NodeIndexTriangle(2, 3, 4).is_ear(&ll));
    assert!(NodeIndexTriangle(3, 4, 1).is_ear(&ll));

    let m = vec![0.0, 0.0, 0.5, 0.5, 1.0, 0.0];
    let (ll, _) = linked_list(&m, 0, m.len(), true);
    assert!(NodeIndexTriangle(3, 1, 2).is_ear(&ll));

    let m = vec![0.0, 0.0, 4.0, 0.0, 4.0, 3.0];
    let (ll, _) = linked_list(&m, 0, m.len(), true);
    assert!(NodeIndexTriangle(3, 1, 2).is_ear(&ll));
}

#[test]
fn test_filter_points() {
    let m = vec![0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    let (mut ll, _) = linked_list(&m, 0, m.len(), true);
    let lllen = ll.nodes.len();
    println!("len {}", ll.nodes.len());
    println!("{}", crate::legacy::dump(&ll));
    let r1 = filter_points(&mut ll, 1, Some(lllen - 1));
    println!("{}", crate::legacy::dump(&ll));
    println!("r1 {} cyclen {}", r1, cycle_len(&ll, r1));
    assert!(cycle_len(&ll, r1) == 4);

    let n = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
    let (mut ll, _) = linked_list(&n, 0, n.len(), true);
    let lllen = ll.nodes.len();
    let r2 = filter_points(&mut ll, 1, Some(lllen - 1));
    assert!(cycle_len(&ll, r2) == 4);

    let n2 = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
    let (mut ll, _) = linked_list(&n2, 0, n2.len(), true);
    let r32 = filter_points(&mut ll, 1, Some(99));
    assert!(cycle_len(&ll, r32) != 4);

    let o = vec![0.0, 0.0, 0.25, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0, 1.0, 0.5, 0.5];
    let (mut ll, _) = linked_list(&o, 0, o.len(), true);
    let lllen = ll.nodes.len();
    let r3 = filter_points(&mut ll, 1, Some(lllen - 1));
    assert!(cycle_len(&ll, r3) == 3);

    let o = vec![0.0, 0.0, 0.5, 0.5, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
    let (mut ll, _) = linked_list(&o, 0, o.len(), true);
    let lllen = ll.nodes.len();
    let r3 = filter_points(&mut ll, 1, Some(lllen - 1));
    assert!(cycle_len(&ll, r3) == 5);
}

#[test]
fn test_earcut_linked() {
    let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    let (mut ll, _) = linked_list(&m, 0, m.len(), true);
    let (mut tris, pass) = (FinalTriangleIndices::default(), 0);
    earcut_linked_hashed(&mut ll, 1, &mut tris, pass);
    assert!(tris.0.len() == 6);

    let m = vec![0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    let (mut ll, _) = linked_list(&m, 0, m.len(), true);
    let (mut tris, pass) = (FinalTriangleIndices::default(), 0);
    earcut_linked_unhashed(&mut ll, 1, &mut tris, pass);
    assert!(tris.0.len() == 9);

    let m = vec![0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    let (mut ll, _) = linked_list(&m, 0, m.len(), true);
    let (mut tris, pass) = (FinalTriangleIndices::default(), 0);
    earcut_linked_hashed(&mut ll, 1, &mut tris, pass);
    assert!(tris.0.len() == 9);
}

#[test]
fn test_middle_inside() {
    let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    let (ll, _) = linked_list(&m, 0, m.len(), true);
    assert!(middle_inside(&ll, &ll.nodes[1], &ll.nodes[3]));
    assert!(middle_inside(&ll, &ll.nodes[2], &ll.nodes[4]));

    let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.9, 0.1];
    let (ll, _) = linked_list(&m, 0, m.len(), true);
    assert!(!middle_inside(&ll, &ll.nodes[1], &ll.nodes[3]));
    assert!(middle_inside(&ll, &ll.nodes[2], &ll.nodes[4]));
}

#[test]
fn test_locally_inside() {
    let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    let (ll, _) = linked_list(&m, 0, m.len(), true);
    assert!(locally_inside(&ll, &ll.nodes[1], &ll.nodes[1]));
    assert!(locally_inside(&ll, &ll.nodes[1], &ll.nodes[2]));
    assert!(locally_inside(&ll, &ll.nodes[1], &ll.nodes[3]));
    assert!(locally_inside(&ll, &ll.nodes[1], &ll.nodes[4]));

    let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.9, 0.1];
    let (ll, _) = linked_list(&m, 0, m.len(), true);
    assert!(locally_inside(&ll, &ll.nodes[1], &ll.nodes[1]));
    assert!(locally_inside(&ll, &ll.nodes[1], &ll.nodes[2]));
    assert!(!locally_inside(&ll, &ll.nodes[1], &ll.nodes[3]));
    assert!(locally_inside(&ll, &ll.nodes[1], &ll.nodes[4]));
}

#[test]
fn test_intersects_polygon() {
    let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    let (ll, _) = linked_list(&m, 0, m.len(), true);

    assert!(!intersects_polygon(&ll, ll.nodes[0], ll.nodes[2]));
    assert!(!intersects_polygon(&ll, ll.nodes[2], ll.nodes[0]));
    assert!(!intersects_polygon(&ll, ll.nodes[1], ll.nodes[3]));
    assert!(!intersects_polygon(&ll, ll.nodes[3], ll.nodes[1]));

    let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.1, 0.1, 0.9, 1.0, 0.0, 1.0];
    let (ll, _) = linked_list(&m, 0, m.len(), true);
    dlog!(9, "{}", crate::legacy::dump(&ll));
    dlog!(5, "{}", intersects_polygon(&ll, ll.nodes[0], ll.nodes[2]));
    dlog!(5, "{}", intersects_polygon(&ll, ll.nodes[2], ll.nodes[0]));
}

#[test]
fn test_intersects_itself() {
    let m = vec![0.0, 0.0, 1.0, 0.0, 0.9, 0.9, 0.0, 1.0];
    let (ll, _) = linked_list(&m, 0, m.len(), true);
    macro_rules! ti {
        ($ok:expr,$a:expr,$b:expr,$c:expr,$d:expr) => {
            assert!(
                $ok == pseudo_intersects(ll.nodes[$a], ll.nodes[$b], ll.nodes[$c], ll.nodes[$d])
            );
        };
    }
    ti!(false, 1, 2 + 1, 1, 1 + 1);
    ti!(false, 1, 2 + 1, 1 + 1, 2 + 1);
    ti!(false, 1, 2 + 1, 2 + 1, 3 + 1);
    ti!(false, 1, 2 + 1, 3 + 1, 1);
    ti!(true, 1, 2 + 1, 3 + 1, 1 + 1);
    ti!(true, 1, 2 + 1, 1 + 1, 3 + 1);
    ti!(true, 2 + 1, 1, 3 + 1, 1 + 1);
    ti!(true, 2 + 1, 1, 1 + 1, 3 + 1);
    ti!(false, 1, 1 + 1, 2 + 1, 3 + 1);
    ti!(false, 1 + 1, 1, 2 + 1, 3 + 1);
    ti!(false, 1, 1, 2 + 1, 3 + 1);
    ti!(false, 1, 1 + 1, 3 + 1, 2 + 1);
    ti!(false, 1 + 1, 1, 3 + 1, 2 + 1);

    ti!(true, 1, 2 + 1, 2 + 1, 1); // special cases
    ti!(true, 1, 2 + 1, 1, 2 + 1);

    let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.1, 0.1, 0.9, 1.0, 0.0, 1.0];
    let (ll, _) = linked_list(&m, 0, m.len(), true);
    assert!(!pseudo_intersects(
        ll.nodes[4],
        ll.nodes[5],
        ll.nodes[1],
        ll.nodes[3]
    ));

    // special case
    assert!(pseudo_intersects(
        ll.nodes[4],
        ll.nodes[5],
        ll.nodes[3],
        ll.nodes[1]
    ));
}

#[test]
fn test_is_valid_diagonal() {
    let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.9, 0.1];
    let (ll, _) = linked_list(&m, 0, m.len(), true);
    assert!(!is_valid_diagonal(&ll, &ll.nodes[1], &ll.nodes[2]));
    assert!(!is_valid_diagonal(&ll, &ll.nodes[2], &ll.nodes[3]));
    assert!(!is_valid_diagonal(&ll, &ll.nodes[3], &ll.nodes[4]));
    assert!(!is_valid_diagonal(&ll, &ll.nodes[4], &ll.nodes[1]));
    assert!(!is_valid_diagonal(&ll, &ll.nodes[1], &ll.nodes[3]));
    assert!(is_valid_diagonal(&ll, &ll.nodes[2], &ll.nodes[4]));
    assert!(!is_valid_diagonal(&ll, &ll.nodes[3], &ll.nodes[4]));
    assert!(is_valid_diagonal(&ll, &ll.nodes[4], &ll.nodes[2]));
}

#[test]
fn test_find_hole_bridge() {
    let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    let (mut ll, _) = linked_list(&m, 0, m.len(), true);
    let hole_idx = ll.insert_node(0, 0.8, 0.8, None);
    assert!(1 == find_hole_bridge(&ll, hole_idx, 1));

    let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.4, 0.5];
    let (mut ll, _) = linked_list(&m, 0, m.len(), true);
    let hole_idx = ll.insert_node(0, 0.5, 0.5, None);
    assert!(5 == find_hole_bridge(&ll, hole_idx, 1));

    let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, -0.4, 0.5];
    let (mut ll, _) = linked_list(&m, 0, m.len(), true);
    let hole_idx = ll.insert_node(0, 0.5, 0.5, None);
    assert!(5 == find_hole_bridge(&ll, hole_idx, 1));

    let m = vec![
        0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, -0.1, 0.9, 0.1, 0.8, -0.1, 0.7, 0.1, 0.6, -0.1, 0.5,
    ];
    let (mut ll, _) = linked_list(&m, 0, m.len(), true);
    let hole_idx = ll.insert_node(0, 0.5, 0.9, None);
    assert!(5 == find_hole_bridge(&ll, hole_idx, 1));
    let hole_idx = ll.insert_node(0, 0.2, 0.1, None);
    assert!(9 == find_hole_bridge(&ll, hole_idx, 1));
    let hole_idx = ll.insert_node(0, 0.2, 0.5, None);
    assert!(9 == find_hole_bridge(&ll, hole_idx, 1));
    let hole_idx = ll.insert_node(0, 0.2, 0.55, None);
    assert!(9 == find_hole_bridge(&ll, hole_idx, 1));
    let hole_idx = ll.insert_node(0, 0.2, 0.6, None);
    assert!(8 == find_hole_bridge(&ll, hole_idx, 1));
    let hole_idx = ll.insert_node(0, 0.2, 0.65, None);
    assert!(7 == find_hole_bridge(&ll, hole_idx, 1));
    let hole_idx = ll.insert_node(0, 0.2, 0.7, None);
    assert!(7 == find_hole_bridge(&ll, hole_idx, 1));
    let hole_idx = ll.insert_node(0, 0.2, 0.75, None);
    assert!(7 == find_hole_bridge(&ll, hole_idx, 1));
    let hole_idx = ll.insert_node(0, 0.2, 0.8, None);
    assert!(6 == find_hole_bridge(&ll, hole_idx, 1));
    let hole_idx = ll.insert_node(0, 0.2, 0.85, None);
    assert!(5 == find_hole_bridge(&ll, hole_idx, 1));
    let hole_idx = ll.insert_node(0, 0.2, 0.9, None);
    assert!(5 == find_hole_bridge(&ll, hole_idx, 1));
    let hole_idx = ll.insert_node(0, 0.2, 0.95, None);
    assert!(5 == find_hole_bridge(&ll, hole_idx, 1));
}

#[test]
fn test_eliminate_hole() {
    let mut body = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];

    let hole = vec![0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9];
    let bodyend = body.len();
    body.extend(hole);
    let holestart = bodyend;
    let holeend = body.len();
    let (mut ll, _) = linked_list(&body, 0, bodyend, true);
    linked_list_add_contour(&mut ll, &body, holestart, holeend, false);
    assert!(cycle_len(&ll, 1) == 4);
    assert!(cycle_len(&ll, 5) == 4);
    eliminate_hole(&mut ll, holestart / DIM + 1, 1);
    println!("{}", crate::legacy::dump(&ll));
    println!("{}", cycle_len(&ll, 1));
    println!("{}", cycle_len(&ll, 7));
    assert!(cycle_len(&ll, 1) == 10);

    let hole = vec![0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8];
    let bodyend = body.len();
    body.extend(hole);
    let holestart = bodyend;
    let holeend = body.len();
    linked_list_add_contour(&mut ll, &body, holestart, holeend, false);
    assert!(cycle_len(&ll, 1) == 10);
    assert!(cycle_len(&ll, 5) == 10);
    assert!(cycle_len(&ll, 11) == 4);
    eliminate_hole(&mut ll, 11, 2);
    assert!(!cycle_len(&ll, 1) != 10);
    assert!(!cycle_len(&ll, 1) != 10);
    assert!(!cycle_len(&ll, 5) != 10);
    assert!(!cycle_len(&ll, 10) != 4);
    assert!(cycle_len(&ll, 1) == 16);
    assert!(cycle_len(&ll, 1) == 16);
    assert!(cycle_len(&ll, 10) == 16);
    assert!(cycle_len(&ll, 15) == 16);
}

#[test]
fn test_cycle_len() {
    let mut body = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.1, 0.1];

    let hole = vec![0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9];
    let bodyend = body.len();
    body.extend(hole);
    let holestart = bodyend;
    let holeend = body.len();
    let (mut ll, _) = linked_list(&body, 0, bodyend, true);
    linked_list_add_contour(&mut ll, &body, holestart, holeend, false);

    let hole = vec![0.2, 0.2, 0.8, 0.2, 0.8, 0.8];
    let bodyend = body.len();
    body.extend(hole);
    let holestart = bodyend;
    let holeend = body.len();
    linked_list_add_contour(&mut ll, &body, holestart, holeend, false);

    dlog!(5, "{}", crate::legacy::dump(&ll));
    dlog!(5, "{}", cycles_report(&ll));
}

#[test]
fn test_cycles_report() {
    let mut body = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.1, 0.1];

    let hole = vec![0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9];
    let bodyend = body.len();
    body.extend(hole);
    let holestart = bodyend;
    let holeend = body.len();
    let (mut ll, _) = linked_list(&body, 0, bodyend, true);
    linked_list_add_contour(&mut ll, &body, holestart, holeend, false);

    let hole = vec![0.2, 0.2, 0.8, 0.2, 0.8, 0.8];
    let bodyend = body.len();
    body.extend(hole);
    let holestart = bodyend;
    let holeend = body.len();
    linked_list_add_contour(&mut ll, &body, holestart, holeend, false);

    dlog!(5, "{}", crate::legacy::dump(&ll));
    dlog!(5, "{}", cycles_report(&ll));
}

#[test]
fn test_eliminate_holes() {
    let mut hole_indices: Vec<usize> = Vec::new();
    let mut body = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
    let (mut ll, _) = linked_list(&body, 0, body.len(), true);
    let hole1 = vec![0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9];
    let hole2 = vec![0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8];
    hole_indices.push(body.len() / DIM);
    hole_indices.push((body.len() + hole1.len()) / DIM);
    body.extend(hole1);
    body.extend(hole2);

    eliminate_holes(&mut ll, &body, &hole_indices, 0);
}

#[test]
fn test_cure_local_intersections() {
    // first test . it will not be able to detect the crossover
    // so it will not change anything.
    let m = vec![
        0.0, 0.0, 1.0, 0.0, 1.1, 0.1, 0.9, 0.1, 1.0, 0.05, 1.0, 1.0, 0.0, 1.0,
    ];
    let (mut ll, _) = linked_list(&m, 0, m.len(), true);
    let mut triangles = FinalTriangleIndices::default();
    cure_local_intersections(&mut ll, 0, &mut triangles);
    assert!(cycle_len(&ll, 1) == 7);
    assert!(triangles.0.is_empty());

    // second test - we have three points that immediately cause
    // self intersection. so it should, in theory, detect and clean
    let m = vec![0.0, 0.0, 1.0, 0.0, 1.1, 0.1, 1.1, 0.0, 1.0, 1.0, 0.0, 1.0];
    let (mut ll, _) = linked_list(&m, 0, m.len(), true);
    let mut triangles = FinalTriangleIndices::default();
    cure_local_intersections(&mut ll, 1, &mut triangles);
    assert!(cycle_len(&ll, 1) == 4);
    assert!(triangles.0.len() == 3);
}

#[test]
fn test_split_earcut() {
    let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];

    let (mut ll, _) = linked_list(&m, 0, m.len(), true);
    let start = 1;
    let mut triangles = FinalTriangleIndices::default();
    split_earcut(&mut ll, start, &mut triangles);
    assert!(triangles.0.len() == 6);
    assert!(ll.nodes.len() == 7);

    let m = vec![
        0.0, 0.0, 1.0, 0.0, 1.5, 0.5, 2.0, 0.0, 3.0, 0.0, 3.0, 1.0, 2.0, 1.0, 1.5, 0.6, 1.0, 1.0,
        0.0, 1.0,
    ];
    let (mut ll, _) = linked_list(&m, 0, m.len(), true);
    let start = 1;
    let mut triangles = FinalTriangleIndices::default();
    split_earcut(&mut ll, start, &mut triangles);
    assert!(ll.nodes.len() == 13);
}

#[test]
fn test_flatten() {
    let data: Vec<Vec<Vec<f64>>> = vec![
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![0.0, 1.0],
        ],
        vec![
            vec![0.1, 0.1],
            vec![0.9, 0.1],
            vec![0.9, 0.9],
            vec![0.1, 0.9],
        ],
        vec![
            vec![0.2, 0.2],
            vec![0.8, 0.2],
            vec![0.8, 0.8],
            vec![0.2, 0.8],
        ],
    ];
    let (coords, hole_indices, dims) = crate::legacy::flatten(&data);
    assert!(DIM == dims);
    println!("{:?} {:?}", coords, hole_indices);
    assert!(coords.len() == 24);
    assert!(hole_indices.len() == 2);
    assert!(hole_indices[0] == 4);
    assert!(hole_indices[1] == 8);
}

#[test]
fn test_iss45() {
    let data = vec![
        vec![
            vec![10.0, 10.0],
            vec![25.0, 10.0],
            vec![25.0, 40.0],
            vec![10.0, 40.0],
        ],
        vec![vec![15.0, 30.0], vec![20.0, 35.0], vec![10.0, 40.0]],
        vec![vec![15.0, 15.0], vec![15.0, 20.0], vec![20.0, 15.0]],
    ];
    let (coords, hole_indices, dims) = crate::legacy::flatten(&data);
    assert!(DIM == dims);
    let triangles = earcut(&coords, &hole_indices, DIM);
    assert!(triangles.len() > 4);
}

#[test]
#[should_panic] // FIXME: This shouldn't panic
fn test_infinite_loop_bug() {
    let coords = [
        3482952.0523706395,
        -2559865.184587028,
        3482952.0523706395,
        -2559865.184587028,
        3856285.4462009706, // HOLE
        -1347264.3952299273,
        3856285.4462009706,
        -1347264.3952299273,
        3864938.7972431043, // HOLE
        -1358303.0608723268,
        3864938.7972431043,
        -1358303.0608723268,
    ];
    let hole_indices = [2, 4];
    earcut(&coords, &hole_indices, DIM);
}
