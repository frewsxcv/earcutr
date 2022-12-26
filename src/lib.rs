use std::{cmp, ops};

static DIM: usize = 2;
static NULL: usize = 0;

#[cfg(test)]
mod tests;

#[doc(hidden)]
pub mod legacy;

pub use legacy::deviation;
pub use legacy::flatten;

type LinkedListNodeIndex = usize;
type VerticesIndex = usize;

pub trait Float: num_traits::float::Float {}

impl<T> Float for T where T: num_traits::float::Float {}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Coord<T: Float> {
    x: T,
    y: T,
}

impl<T: Float> Coord<T> {
    // z-order of a point given coords and inverse of the longer side of
    // data bbox
    #[inline(always)]
    fn zorder(&self, invsize: T) -> i32 {
        // coords are transformed into non-negative 15-bit integer range
        // stored in two 32bit ints, which are combined into a single 64 bit int.
        let x: i64 = num_traits::cast::<T, i64>(self.x * invsize).unwrap();
        let y: i64 = num_traits::cast::<T, i64>(self.y * invsize).unwrap();
        let mut xy: i64 = x << 32 | y;

        // todo ... big endian?
        xy = (xy | (xy << 8)) & 0x00FF00FF00FF00FF;
        xy = (xy | (xy << 4)) & 0x0F0F0F0F0F0F0F0F;
        xy = (xy | (xy << 2)) & 0x3333333333333333;
        xy = (xy | (xy << 1)) & 0x5555555555555555;

        ((xy >> 32) | (xy << 1)) as i32
    }
}

#[derive(Clone, Copy, Debug)]
struct LinkedListNode<T: Float> {
    /// vertex index in flat one-d array of 64bit float coords
    vertices_index: VerticesIndex,
    /// vertex
    coord: Coord<T>,
    /// previous vertex node in a polygon ring
    prev_linked_list_node_index: LinkedListNodeIndex,
    /// next vertex node in a polygon ring
    next_linked_list_node_index: LinkedListNodeIndex,
    /// z-order curve value
    z: i32,
    /// previous node in z-order
    prevz_idx: LinkedListNodeIndex,
    /// next node in z-order
    nextz_idx: LinkedListNodeIndex,
    /// indicates whether this is a steiner point
    is_steiner_point: bool,
    /// index within LinkedLists vector that holds all nodes
    idx: LinkedListNodeIndex,
}

impl<T: Float> LinkedListNode<T> {
    fn new(i: VerticesIndex, coord: Coord<T>, idx: LinkedListNodeIndex) -> LinkedListNode<T> {
        LinkedListNode {
            vertices_index: i,
            coord,
            prev_linked_list_node_index: NULL,
            next_linked_list_node_index: NULL,
            z: 0,
            nextz_idx: NULL,
            prevz_idx: NULL,
            is_steiner_point: false,
            idx,
        }
    }

    // check if two points are equal
    fn xy_eq(&self, other: LinkedListNode<T>) -> bool {
        self.coord == other.coord
    }

    fn prev_linked_list_node(&self, linked_list_nodes: &LinkedLists<T>) -> LinkedListNode<T> {
        linked_list_nodes.nodes[self.prev_linked_list_node_index]
    }

    fn next_linked_list_node(&self, linked_list_nodes: &LinkedLists<T>) -> LinkedListNode<T> {
        linked_list_nodes.nodes[self.next_linked_list_node_index]
    }
}

pub struct LinkedLists<T: Float> {
    nodes: Vec<LinkedListNode<T>>,
    invsize: T,
    min: Coord<T>,
    max: Coord<T>,
    usehash: bool,
}

pub trait Vertices<T: Float> {
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool;

    fn vertex(&self, index: usize) -> T;

    fn signed_area(&self, start: VerticesIndex, end: VerticesIndex) -> T {
        let i = (start..end).step_by(DIM);
        let j = (start..end).cycle().skip((end - DIM) - start).step_by(DIM);
        let zero = T::zero();
        i.zip(j).fold(zero, |s, (i, j)| {
            s + (self.vertex(j) - self.vertex(i)) * (self.vertex(i + 1) + self.vertex(j + 1))
        })
    }
}

// Note: none of the following macros work for Left-Hand-Side of assignment.
macro_rules! next {
    ($ll:expr,$idx:expr) => {
        $ll.nodes[$ll.nodes[$idx].next_linked_list_node_index]
    };
}
macro_rules! nextref {
    ($ll:expr,$idx:expr) => {
        &$ll.nodes[$ll.nodes[$idx].next_linked_list_node_index]
    };
}
macro_rules! prev {
    ($ll:expr,$idx:expr) => {
        $ll.nodes[$ll.nodes[$idx].prev_linked_list_node_index]
    };
}
macro_rules! prevref {
    ($ll:expr,$idx:expr) => {
        &$ll.nodes[$ll.nodes[$idx].prev_linked_list_node_index]
    };
}

impl<T: Float> LinkedLists<T> {
    fn iter(&self, r: ops::Range<LinkedListNodeIndex>) -> NodeIterator<T> {
        NodeIterator::new(self, r.start, r.end)
    }

    fn iter_pairs(&self, r: ops::Range<LinkedListNodeIndex>) -> NodePairIterator<T> {
        NodePairIterator::new(self, r.start, r.end)
    }

    fn insert_node(
        &mut self,
        i: VerticesIndex,
        coord: Coord<T>,
        last: Option<LinkedListNodeIndex>,
    ) -> LinkedListNodeIndex {
        let mut p = LinkedListNode::new(i, coord, self.nodes.len());
        match last {
            None => {
                p.next_linked_list_node_index = p.idx;
                p.prev_linked_list_node_index = p.idx;
            }
            Some(last) => {
                p.next_linked_list_node_index = self.nodes[last].next_linked_list_node_index;
                p.prev_linked_list_node_index = last;
                let lastnextidx = self.nodes[last].next_linked_list_node_index;
                self.nodes[lastnextidx].prev_linked_list_node_index = p.idx;
                self.nodes[last].next_linked_list_node_index = p.idx;
            }
        }
        let result = p.idx;
        self.nodes.push(p);
        result
    }
    fn remove_node(&mut self, p_idx: LinkedListNodeIndex) {
        let pi = self.nodes[p_idx].prev_linked_list_node_index;
        let ni = self.nodes[p_idx].next_linked_list_node_index;
        let pz = self.nodes[p_idx].prevz_idx;
        let nz = self.nodes[p_idx].nextz_idx;
        self.nodes[pi].next_linked_list_node_index = ni;
        self.nodes[ni].prev_linked_list_node_index = pi;
        self.nodes[pz].nextz_idx = nz;
        self.nodes[nz].prevz_idx = pz;
    }
    fn new(size_hint: usize) -> LinkedLists<T> {
        let mut ll = LinkedLists {
            nodes: Vec::with_capacity(size_hint),
            invsize: T::zero(),
            min: Coord {
                x: T::max_value(),
                y: T::max_value(),
            },
            max: Coord {
                x: T::min_value(),
                y: T::min_value(),
            },
            usehash: true,
        };
        // ll.nodes[0] is the NULL node. For example usage, see remove_node()
        ll.nodes.push(LinkedListNode {
            vertices_index: 0,
            coord: Coord {
                x: T::zero(),
                y: T::zero(),
            },
            prev_linked_list_node_index: 0,
            next_linked_list_node_index: 0,
            z: 0,
            nextz_idx: 0,
            prevz_idx: 0,
            is_steiner_point: false,
            idx: 0,
        });
        ll
    }

    // interlink polygon nodes in z-order
    fn index_curve(&mut self, start: LinkedListNodeIndex) {
        let invsize = self.invsize;
        let mut p = start;
        loop {
            if self.nodes[p].z == 0 {
                self.nodes[p].z = self.nodes[p].coord.zorder(invsize);
            }
            self.nodes[p].prevz_idx = self.nodes[p].prev_linked_list_node_index;
            self.nodes[p].nextz_idx = self.nodes[p].next_linked_list_node_index;
            p = self.nodes[p].next_linked_list_node_index;
            if p == start {
                break;
            }
        }

        let pzi = self.nodes[start].prevz_idx;
        self.nodes[pzi].nextz_idx = NULL;
        self.nodes[start].prevz_idx = NULL;
        self.sort_linked(start);
    }

    // find a bridge between vertices that connects hole with an outer ring
    // and and link it
    fn eliminate_hole(
        &mut self,
        hole_idx: LinkedListNodeIndex,
        outer_node_idx: LinkedListNodeIndex,
    ) {
        let test_idx = find_hole_bridge(self, hole_idx, outer_node_idx);
        let b = split_bridge_polygon(self, test_idx, hole_idx);
        let ni = self.nodes[b].next_linked_list_node_index;
        filter_points(self, b, Some(ni));
    }

    // Simon Tatham's linked list merge sort algorithm
    // http://www.chiark.greenend.org.uk/~sgtatham/algorithms/listsort.html
    fn sort_linked(&mut self, mut list: LinkedListNodeIndex) {
        let mut p;
        let mut q;
        let mut e;
        let mut nummerges;
        let mut psize;
        let mut qsize;
        let mut insize = 1;
        let mut tail;

        loop {
            p = list;
            list = NULL;
            tail = NULL;
            nummerges = 0;

            while p != NULL {
                nummerges += 1;
                q = p;
                psize = 0;
                while q != NULL && psize < insize {
                    psize += 1;
                    q = self.nodes[q].nextz_idx;
                }
                qsize = insize;

                while psize > 0 || (qsize > 0 && q != NULL) {
                    if psize > 0 && (qsize == 0 || q == NULL || self.nodes[p].z <= self.nodes[q].z)
                    {
                        e = p;
                        p = self.nodes[p].nextz_idx;
                        psize -= 1;
                    } else {
                        e = q;
                        q = self.nodes[q].nextz_idx;
                        qsize -= 1;
                    }

                    if tail != NULL {
                        self.nodes[tail].nextz_idx = e;
                    } else {
                        list = e;
                    }

                    self.nodes[e].prevz_idx = tail;
                    tail = e;
                }

                p = q;
            }

            self.nodes[tail].nextz_idx = NULL;
            insize *= 2;
            if nummerges <= 1 {
                break;
            }
        }
    }

    // add new nodes to an existing linked list.
    fn add_contour<V: Vertices<T>>(
        &mut self,
        vertices: &V,
        start: VerticesIndex,
        end: VerticesIndex,
        clockwise: bool,
    ) -> (LinkedListNodeIndex, LinkedListNodeIndex) {
        assert!(start <= vertices.len() && end <= vertices.len() && !vertices.is_empty());
        // Previous code:
        //
        // if start > vertices.len() || end > vertices.len() || vertices.is_empty() {
        //     return (None, None);
        // }
        let mut lastidx = None;
        let mut leftmost_idx = None;
        let mut contour_minx = T::max_value();

        if clockwise == (vertices.signed_area(start, end) > T::zero()) {
            for i in (start..end).step_by(DIM) {
                lastidx = Some(self.insert_node(
                    i / DIM,
                    Coord {
                        x: vertices.vertex(i),
                        y: vertices.vertex(i + 1),
                    },
                    lastidx,
                ));
                if contour_minx > vertices.vertex(i) {
                    contour_minx = vertices.vertex(i);
                    leftmost_idx = lastidx
                };
                if self.usehash {
                    self.min.y = vertices.vertex(i + 1).min(self.min.y);
                    self.max.x = vertices.vertex(i).max(self.max.x);
                    self.max.y = vertices.vertex(i + 1).max(self.max.y);
                }
            }
        } else {
            for i in (start..=(end - DIM)).rev().step_by(DIM) {
                lastidx = Some(self.insert_node(
                    i / DIM,
                    Coord {
                        x: vertices.vertex(i),
                        y: vertices.vertex(i + 1),
                    },
                    lastidx,
                ));
                if contour_minx > vertices.vertex(i) {
                    contour_minx = vertices.vertex(i);
                    leftmost_idx = lastidx
                };
                if self.usehash {
                    self.min.y = vertices.vertex(i + 1).min(self.min.y);
                    self.max.x = vertices.vertex(i).max(self.max.x);
                    self.max.y = vertices.vertex(i + 1).max(self.max.y);
                }
            }
        }

        self.min.x = contour_minx.min(self.min.x);

        if self.nodes[lastidx.unwrap()].xy_eq(*nextref!(self, lastidx.unwrap())) {
            self.remove_node(lastidx.unwrap());
            lastidx = Some(self.nodes[lastidx.unwrap()].next_linked_list_node_index);
        }
        (lastidx.unwrap(), leftmost_idx.unwrap())
    }

    // check if a diagonal between two polygon nodes is valid (lies in
    // polygon interior)
    fn is_valid_diagonal(&self, a: &LinkedListNode<T>, b: &LinkedListNode<T>) -> bool {
        next!(self, a.idx).vertices_index != b.vertices_index
            && prev!(self, a.idx).vertices_index != b.vertices_index
            && !intersects_polygon(self, *a, *b)
            && locally_inside(self, a, b)
            && locally_inside(self, b, a)
            && middle_inside(self, a, b)
    }
}

struct NodeIterator<'a, T: Float> {
    cur: LinkedListNodeIndex,
    end: LinkedListNodeIndex,
    ll: &'a LinkedLists<T>,
    pending_result: Option<&'a LinkedListNode<T>>,
}

impl<'a, T: Float> NodeIterator<'a, T> {
    fn new(
        ll: &LinkedLists<T>,
        start: LinkedListNodeIndex,
        end: LinkedListNodeIndex,
    ) -> NodeIterator<T> {
        NodeIterator {
            pending_result: Some(&ll.nodes[start]),
            cur: start,
            end,
            ll,
        }
    }
}

impl<'a, T: Float> Iterator for NodeIterator<'a, T> {
    type Item = &'a LinkedListNode<T>;
    fn next(&mut self) -> Option<Self::Item> {
        self.cur = self.ll.nodes[self.cur].next_linked_list_node_index;
        let cur_result = self.pending_result;
        self.pending_result = if self.cur == self.end {
            // only one branch, saves time
            None
        } else {
            Some(&self.ll.nodes[self.cur])
        };
        cur_result
    }
}

struct NodePairIterator<'a, T: Float> {
    cur: LinkedListNodeIndex,
    end: LinkedListNodeIndex,
    ll: &'a LinkedLists<T>,
    pending_result: Option<(&'a LinkedListNode<T>, &'a LinkedListNode<T>)>,
}

impl<'a, T: Float> NodePairIterator<'a, T> {
    fn new(
        ll: &LinkedLists<T>,
        start: LinkedListNodeIndex,
        end: LinkedListNodeIndex,
    ) -> NodePairIterator<T> {
        NodePairIterator {
            pending_result: Some((&ll.nodes[start], nextref!(ll, start))),
            cur: start,
            end,
            ll,
        }
    }
}

impl<'a, T: Float> Iterator for NodePairIterator<'a, T> {
    type Item = (&'a LinkedListNode<T>, &'a LinkedListNode<T>);
    fn next(&mut self) -> Option<Self::Item> {
        self.cur = self.ll.nodes[self.cur].next_linked_list_node_index;
        let cur_result = self.pending_result;
        if self.cur == self.end {
            // only one branch, saves time
            self.pending_result = None;
        } else {
            self.pending_result = Some((&self.ll.nodes[self.cur], nextref!(self.ll, self.cur)))
        }
        cur_result
    }
}

// link every hole into the outer loop, producing a single-ring polygon
// without holes
fn eliminate_holes<T: Float, V: Vertices<T>>(
    ll: &mut LinkedLists<T>,
    vertices: &V,
    hole_indices: &[VerticesIndex],
    inouter_node: LinkedListNodeIndex,
) -> LinkedListNodeIndex {
    let mut outer_node = inouter_node;
    let mut queue: Vec<LinkedListNode<T>> = Vec::new();
    for i in 0..hole_indices.len() {
        let vertices_hole_start_index = hole_indices[i] * DIM;
        let vertices_hole_end_index = if i < (hole_indices.len() - 1) {
            hole_indices[i + 1] * DIM
        } else {
            vertices.len()
        };
        let (list, leftmost_idx) = ll.add_contour(
            vertices,
            vertices_hole_start_index,
            vertices_hole_end_index,
            false,
        );
        if list == ll.nodes[list].next_linked_list_node_index {
            ll.nodes[list].is_steiner_point = true;
        }
        queue.push(ll.nodes[leftmost_idx]);
    }

    queue.sort_by(|a, b| {
        a.coord
            .x
            .partial_cmp(&b.coord.x)
            .unwrap_or(cmp::Ordering::Equal)
    });

    // process holes from left to right
    for node in queue {
        ll.eliminate_hole(node.idx, outer_node);
        let nextidx = next!(ll, outer_node).idx;
        outer_node = filter_points(ll, outer_node, Some(nextidx));
    }
    outer_node
} // elim holes

impl<const N: usize, T: Float> Vertices<T> for [T; N] {
    fn len(&self) -> usize {
        <[T]>::len(self)
    }

    fn is_empty(&self) -> bool {
        <[T]>::is_empty(self)
    }

    fn vertex(&self, index: usize) -> T {
        self[index]
    }
}

impl<T: Float> Vertices<T> for Vec<T> {
    fn len(&self) -> usize {
        <Vec<T>>::len(self)
    }

    fn is_empty(&self) -> bool {
        <Vec<T>>::is_empty(self)
    }

    fn vertex(&self, index: usize) -> T {
        self[index]
    }
}

// minx, miny and invsize are later used to transform coords
// into integers for z-order calculation
fn calc_invsize<T: Float>(min: Coord<T>, max: Coord<T>) -> T {
    let invsize = (max.x - min.x).max(max.y - min.y);
    match invsize.is_zero() {
        true => T::zero(),
        false => num_traits::cast::<f64, T>(32767.0).unwrap() / invsize,
    }
}

// main ear slicing loop which triangulates a polygon (given as a linked
// list)
fn earcut_linked_hashed<const PASS: usize, T: Float>(
    ll: &mut LinkedLists<T>,
    mut ear_idx: LinkedListNodeIndex,
    triangle_indices: &mut FinalTriangleIndices,
) {
    // interlink polygon nodes in z-order
    if PASS == 0 {
        ll.index_curve(ear_idx);
    }
    // iterate through ears, slicing them one by one
    let mut stop_idx = ear_idx;
    let mut prev_idx = 0;
    let mut next_idx = ll.nodes[ear_idx].next_linked_list_node_index;
    while stop_idx != next_idx {
        prev_idx = ll.nodes[ear_idx].prev_linked_list_node_index;
        next_idx = ll.nodes[ear_idx].next_linked_list_node_index;
        let node_index_triangle = NodeIndexTriangle(prev_idx, ear_idx, next_idx);
        if node_index_triangle.node_triangle(ll).is_ear_hashed(ll) {
            triangle_indices.push(VerticesIndexTriangle(
                ll.nodes[prev_idx].vertices_index,
                ll.nodes[ear_idx].vertices_index,
                ll.nodes[next_idx].vertices_index,
            ));
            ll.remove_node(ear_idx);
            // skipping the next vertex leads to less sliver triangles
            ear_idx = ll.nodes[next_idx].next_linked_list_node_index;
            stop_idx = ear_idx;
        } else {
            ear_idx = next_idx;
        }
    }

    if prev_idx == next_idx {
        return;
    };
    // if we looped through the whole remaining polygon and can't
    // find any more ears
    if PASS == 0 {
        let tmp = filter_points(ll, next_idx, None);
        earcut_linked_hashed::<1, T>(ll, tmp, triangle_indices);
    } else if PASS == 1 {
        ear_idx = cure_local_intersections(ll, next_idx, triangle_indices);
        earcut_linked_hashed::<2, T>(ll, ear_idx, triangle_indices);
    } else if PASS == 2 {
        split_earcut(ll, next_idx, triangle_indices);
    }
}

// main ear slicing loop which triangulates a polygon (given as a linked
// list)
fn earcut_linked_unhashed<const PASS: usize, T: Float>(
    ll: &mut LinkedLists<T>,
    mut ear_idx: LinkedListNodeIndex,
    triangles: &mut FinalTriangleIndices,
) {
    // iterate through ears, slicing them one by one
    let mut stop_idx = ear_idx;
    let mut prev_idx = 0;
    let mut next_idx = ll.nodes[ear_idx].next_linked_list_node_index;
    while stop_idx != next_idx {
        prev_idx = ll.nodes[ear_idx].prev_linked_list_node_index;
        next_idx = ll.nodes[ear_idx].next_linked_list_node_index;
        if NodeIndexTriangle(prev_idx, ear_idx, next_idx).is_ear(ll) {
            triangles.push(VerticesIndexTriangle(
                ll.nodes[prev_idx].vertices_index,
                ll.nodes[ear_idx].vertices_index,
                ll.nodes[next_idx].vertices_index,
            ));
            ll.remove_node(ear_idx);
            // skipping the next vertex leads to less sliver triangles
            ear_idx = ll.nodes[next_idx].next_linked_list_node_index;
            stop_idx = ear_idx;
        } else {
            ear_idx = next_idx;
        }
    }

    if prev_idx == next_idx {
        return;
    };
    // if we looped through the whole remaining polygon and can't
    // find any more ears
    if PASS == 0 {
        let tmp = filter_points(ll, next_idx, None);
        earcut_linked_unhashed::<1, T>(ll, tmp, triangles);
    } else if PASS == 1 {
        ear_idx = cure_local_intersections(ll, next_idx, triangles);
        earcut_linked_unhashed::<2, T>(ll, ear_idx, triangles);
    } else if PASS == 2 {
        split_earcut(ll, next_idx, triangles);
    }
}

#[derive(Clone, Copy)]
struct NodeIndexTriangle(
    LinkedListNodeIndex,
    LinkedListNodeIndex,
    LinkedListNodeIndex,
);

impl NodeIndexTriangle {
    fn prev_node<T: Float>(self, ll: &LinkedLists<T>) -> LinkedListNode<T> {
        ll.nodes[self.0]
    }

    fn ear_node<T: Float>(self, ll: &LinkedLists<T>) -> LinkedListNode<T> {
        ll.nodes[self.1]
    }

    fn next_node<T: Float>(self, ll: &LinkedLists<T>) -> LinkedListNode<T> {
        ll.nodes[self.2]
    }

    fn node_triangle<T: Float>(self, ll: &LinkedLists<T>) -> NodeTriangle<T> {
        NodeTriangle(self.prev_node(ll), self.ear_node(ll), self.next_node(ll))
    }

    fn area<T: Float>(self, ll: &LinkedLists<T>) -> T {
        self.node_triangle(ll).area()
    }

    // check whether a polygon node forms a valid ear with adjacent nodes
    fn is_ear<T: Float>(self, ll: &LinkedLists<T>) -> bool {
        let zero = T::zero();
        match self.area(ll) >= zero {
            true => false, // reflex, cant be ear
            false => !ll
                .iter(self.next_node(ll).next_linked_list_node_index..self.prev_node(ll).idx)
                .any(|p| {
                    self.node_triangle(ll).contains_point(*p)
                        && (NodeTriangle(*prevref!(ll, p.idx), *p, *nextref!(ll, p.idx)).area()
                            >= zero)
                }),
        }
    }
}

#[derive(Clone, Copy)]
struct NodeTriangle<T: Float>(LinkedListNode<T>, LinkedListNode<T>, LinkedListNode<T>);

impl<T: Float> NodeTriangle<T> {
    fn from_ear_node(ear_node: LinkedListNode<T>, ll: &mut LinkedLists<T>) -> Self {
        NodeTriangle(
            ear_node.prev_linked_list_node(ll),
            ear_node,
            ear_node.next_linked_list_node(ll),
        )
    }

    fn area(&self) -> T {
        let p = self.0;
        let q = self.1;
        let r = self.2;
        // signed area of a parallelogram
        (q.coord.y - p.coord.y) * (r.coord.x - q.coord.x)
            - (q.coord.x - p.coord.x) * (r.coord.y - q.coord.y)
    }

    // check if a point lies within a convex triangle
    fn contains_point(&self, p: LinkedListNode<T>) -> bool {
        let zero = T::zero();

        ((self.2.coord.x - p.coord.x) * (self.0.coord.y - p.coord.y)
            - (self.0.coord.x - p.coord.x) * (self.2.coord.y - p.coord.y)
            >= zero)
            && ((self.0.coord.x - p.coord.x) * (self.1.coord.y - p.coord.y)
                - (self.1.coord.x - p.coord.x) * (self.0.coord.y - p.coord.y)
                >= zero)
            && ((self.1.coord.x - p.coord.x) * (self.2.coord.y - p.coord.y)
                - (self.2.coord.x - p.coord.x) * (self.1.coord.y - p.coord.y)
                >= zero)
    }

    #[inline(always)]
    fn is_ear_hashed(&self, ll: &mut LinkedLists<T>) -> bool {
        let zero = T::zero();

        if self.area() >= zero {
            return false;
        };
        let NodeTriangle(prev, ear, next) = self;

        let bbox_maxx = prev.coord.x.max(ear.coord.x.max(next.coord.x));
        let bbox_maxy = prev.coord.y.max(ear.coord.y.max(next.coord.y));
        let bbox_minx = prev.coord.x.min(ear.coord.x.min(next.coord.x));
        let bbox_miny = prev.coord.y.min(ear.coord.y.min(next.coord.y));
        // z-order range for the current triangle bbox;
        let min_z = Coord {
            x: bbox_minx,
            y: bbox_miny,
        }
        .zorder(ll.invsize);
        let max_z = Coord {
            x: bbox_maxx,
            y: bbox_maxy,
        }
        .zorder(ll.invsize);

        let mut p = ear.prevz_idx;
        let mut n = ear.nextz_idx;
        while (p != NULL) && (ll.nodes[p].z >= min_z) && (n != NULL) && (ll.nodes[n].z <= max_z) {
            if earcheck(
                prev,
                ear,
                next,
                prevref!(ll, p),
                &ll.nodes[p],
                nextref!(ll, p),
            ) {
                return false;
            }
            p = ll.nodes[p].prevz_idx;

            if earcheck(
                prev,
                ear,
                next,
                prevref!(ll, n),
                &ll.nodes[n],
                nextref!(ll, n),
            ) {
                return false;
            }
            n = ll.nodes[n].nextz_idx;
        }

        ll.nodes[NULL].z = min_z - 1;
        while ll.nodes[p].z >= min_z {
            if earcheck(
                prev,
                ear,
                next,
                prevref!(ll, p),
                &ll.nodes[p],
                nextref!(ll, p),
            ) {
                return false;
            }
            p = ll.nodes[p].prevz_idx;
        }

        ll.nodes[NULL].z = max_z + 1;
        while ll.nodes[n].z <= max_z {
            if earcheck(
                prev,
                ear,
                next,
                prevref!(ll, n),
                &ll.nodes[n],
                nextref!(ll, n),
            ) {
                return false;
            }
            n = ll.nodes[n].nextz_idx;
        }

        true
    }
}

// helper for is_ear_hashed. needs manual inline (rust 2018)
#[inline(always)]
fn earcheck<T: Float>(
    a: &LinkedListNode<T>,
    b: &LinkedListNode<T>,
    c: &LinkedListNode<T>,
    prev: &LinkedListNode<T>,
    p: &LinkedListNode<T>,
    next: &LinkedListNode<T>,
) -> bool {
    let zero = T::zero();

    (p.idx != a.idx)
        && (p.idx != c.idx)
        && NodeTriangle(*a, *b, *c).contains_point(*p)
        && NodeTriangle(*prev, *p, *next).area() >= zero
}

fn filter_points<T: Float>(
    ll: &mut LinkedLists<T>,
    start: LinkedListNodeIndex,
    end: Option<LinkedListNodeIndex>,
) -> LinkedListNodeIndex {
    let mut end = end.unwrap_or(start);
    if end >= ll.nodes.len() || start >= ll.nodes.len() {
        return NULL;
    }

    let mut p = start;
    let mut again;

    // this loop "wastes" calculations by going over the same points multiple
    // times. however, altering the location of the 'end' node can disrupt
    // the algorithm of other code that calls the filter_points function.
    loop {
        again = false;
        if !ll.nodes[p].is_steiner_point
            && (ll.nodes[p].xy_eq(ll.nodes[ll.nodes[p].next_linked_list_node_index])
                || NodeTriangle::from_ear_node(ll.nodes[p], ll)
                    .area()
                    .is_zero())
        {
            ll.remove_node(p);
            end = ll.nodes[p].prev_linked_list_node_index;
            p = end;
            if p == ll.nodes[p].next_linked_list_node_index {
                break end;
            }
            again = true;
        } else {
            if p == ll.nodes[p].next_linked_list_node_index {
                break NULL;
            }
            p = ll.nodes[p].next_linked_list_node_index;
        }
        if !again && p == end {
            break end;
        }
    }
}

// create a circular doubly linked list from polygon points in the
// specified winding order
fn linked_list<T: Float, V: Vertices<T>>(
    vertices: &V,
    start: usize,
    end: usize,
    clockwise: bool,
) -> (LinkedLists<T>, LinkedListNodeIndex) {
    let mut ll: LinkedLists<T> = LinkedLists::new(vertices.len() / DIM);
    if vertices.len() < 80 {
        ll.usehash = false
    };
    let (last_idx, _) = ll.add_contour(vertices, start, end, clockwise);
    (ll, last_idx)
}

struct VerticesIndexTriangle(usize, usize, usize);

#[derive(Default, Debug)]
struct FinalTriangleIndices(Vec<usize>);

impl FinalTriangleIndices {
    fn push(&mut self, vertices_index_triangle: VerticesIndexTriangle) {
        self.0.push(vertices_index_triangle.0);
        self.0.push(vertices_index_triangle.1);
        self.0.push(vertices_index_triangle.2);
    }
}

pub fn earcut<T: Float, V: Vertices<T>>(
    vertices: &V,
    hole_indices: &[VerticesIndex],
    dims: usize,
) -> Vec<usize> {
    if vertices.is_empty() {
        return vec![];
    }

    let outer_len = match hole_indices.len() {
        0 => vertices.len(),
        _ => hole_indices[0] * DIM,
    };

    let (mut ll, outer_node) = linked_list(vertices, 0, outer_len, true);
    let mut triangles = FinalTriangleIndices(Vec::with_capacity(vertices.len() / DIM));
    if ll.nodes.len() == 1 || DIM != dims {
        return triangles.0;
    }

    let outer_node = eliminate_holes(&mut ll, vertices, hole_indices, outer_node);

    if ll.usehash {
        ll.invsize = calc_invsize(ll.min, ll.max);

        // translate all points so min is 0,0. prevents subtraction inside
        // zorder. also note invsize does not depend on translation in space
        // if one were translating in a space with an even spaced grid of points.
        // floating point space is not evenly spaced, but it is close enough for
        // this hash algorithm
        let (mx, my) = (ll.min.x, ll.min.y);
        ll.nodes.iter_mut().for_each(|n| {
            n.coord.x = n.coord.x - mx;
            n.coord.y = n.coord.y - my;
        });
        earcut_linked_hashed::<0, T>(&mut ll, outer_node, &mut triangles);
    } else {
        earcut_linked_unhashed::<0, T>(&mut ll, outer_node, &mut triangles);
    }

    triangles.0
}

/* go through all polygon nodes and cure small local self-intersections
what is a small local self-intersection? well, lets say you have four points
a,b,c,d. now imagine you have three line segments, a-b, b-c, and c-d. now
imagine two of those segments overlap each other. thats an intersection. so
this will remove one of those nodes so there is no more overlap.

but theres another important aspect of this function. it will dump triangles
into the 'triangles' variable, thus this is part of the triangulation
algorithm itself.*/
fn cure_local_intersections<T: Float>(
    ll: &mut LinkedLists<T>,
    instart: LinkedListNodeIndex,
    triangles: &mut FinalTriangleIndices,
) -> LinkedListNodeIndex {
    let mut p = instart;
    let mut start = instart;

    //        2--3  4--5 << 2-3 + 4-5 pseudointersects
    //           x  x
    //  0  1  2  3  4  5  6  7
    //  a  p  pn b
    //              eq     a      b
    //              psi    a p pn b
    //              li  pa a p pn b bn
    //              tp     a p    b
    //              rn       p pn
    //              nst    a      p pn b
    //                            st

    //
    //                            a p  pn b

    loop {
        let a = ll.nodes[p].prev_linked_list_node_index;
        let b = next!(ll, p).next_linked_list_node_index;

        if !ll.nodes[a].xy_eq(ll.nodes[b])
            && pseudo_intersects(
                ll.nodes[a],
                ll.nodes[p],
                *nextref!(ll, p),
                ll.nodes[b],
            )
			// prev next a, prev next b
            && locally_inside(ll, &ll.nodes[a], &ll.nodes[b])
            && locally_inside(ll, &ll.nodes[b], &ll.nodes[a])
        {
            triangles.push(VerticesIndexTriangle(
                ll.nodes[a].vertices_index,
                ll.nodes[p].vertices_index,
                ll.nodes[b].vertices_index,
            ));

            // remove two nodes involved
            ll.remove_node(p);
            let nidx = ll.nodes[p].next_linked_list_node_index;
            ll.remove_node(nidx);

            start = ll.nodes[b].idx;
            p = start;
        }
        p = ll.nodes[p].next_linked_list_node_index;
        if p == start {
            break;
        }
    }

    p
}

// try splitting polygon into two and triangulate them independently
fn split_earcut<T: Float>(
    ll: &mut LinkedLists<T>,
    start_idx: LinkedListNodeIndex,
    triangles: &mut FinalTriangleIndices,
) {
    // look for a valid diagonal that divides the polygon into two
    let mut a = start_idx;
    loop {
        let mut b = next!(ll, a).next_linked_list_node_index;
        while b != ll.nodes[a].prev_linked_list_node_index {
            if ll.nodes[a].vertices_index != ll.nodes[b].vertices_index
                && ll.is_valid_diagonal(&ll.nodes[a], &ll.nodes[b])
            {
                // split the polygon in two by the diagonal
                let mut c = split_bridge_polygon(ll, a, b);

                // filter colinear points around the cuts
                let an = ll.nodes[a].next_linked_list_node_index;
                let cn = ll.nodes[c].next_linked_list_node_index;
                a = filter_points(ll, a, Some(an));
                c = filter_points(ll, c, Some(cn));

                // run earcut on each half
                earcut_linked_hashed::<0, T>(ll, a, triangles);
                earcut_linked_hashed::<0, T>(ll, c, triangles);
                return;
            }
            b = ll.nodes[b].next_linked_list_node_index;
        }
        a = ll.nodes[a].next_linked_list_node_index;
        if a == start_idx {
            break;
        }
    }
}

// David Eberly's algorithm for finding a bridge between hole and outer polygon
fn find_hole_bridge<T: Float>(
    ll: &LinkedLists<T>,
    hole: LinkedListNodeIndex,
    outer_node: LinkedListNodeIndex,
) -> LinkedListNodeIndex {
    let mut p = outer_node;
    let hx = ll.nodes[hole].coord.x;
    let hy = ll.nodes[hole].coord.y;
    let mut qx = T::neg_infinity();
    let mut m: Option<LinkedListNodeIndex> = None;

    // find a segment intersected by a ray from the hole's leftmost
    // point to the left; segment's endpoint with lesser x will be
    // potential connection point
    let calcx = |p: &LinkedListNode<T>| {
        p.coord.x
            + (hy - p.coord.y) * (next!(ll, p.idx).coord.x - p.coord.x)
                / (next!(ll, p.idx).coord.y - p.coord.y)
    };
    for (p, n) in ll
        .iter_pairs(p..outer_node)
        .filter(|(p, n)| hy <= p.coord.y && hy >= n.coord.y)
        .filter(|(p, n)| n.coord.y != p.coord.y)
        .filter(|(p, _)| calcx(p) <= hx)
    {
        if qx < calcx(p) {
            qx = calcx(p);
            if qx == hx && hy == p.coord.y {
                return p.idx;
            } else if qx == hx && hy == n.coord.y {
                return p.next_linked_list_node_index;
            }
            m = if p.coord.x < n.coord.x {
                Some(p.idx)
            } else {
                Some(n.idx)
            };
        }
    }

    let Some(m) = m else { return NULL };

    // hole touches outer segment; pick lower endpoint
    if hx == qx {
        return prev!(ll, m).idx;
    }

    // look for points inside the triangle of hole point, segment
    // intersection and endpoint; if there are no points found, we have
    // a valid connection; otherwise choose the point of the minimum
    // angle with the ray as connection point

    let mp = LinkedListNode::new(0, ll.nodes[m].coord, 0);
    p = next!(ll, m).idx;
    let x1 = if hy < mp.coord.y { hx } else { qx };
    let x2 = if hy < mp.coord.y { qx } else { hx };
    let n1 = LinkedListNode::new(0, Coord { x: x1, y: hy }, 0);
    let n2 = LinkedListNode::new(0, Coord { x: x2, y: hy }, 0);
    let two = num_traits::cast::<f64, T>(2.).unwrap();

    let calctan = |p: &LinkedListNode<T>| (hy - p.coord.y).abs() / (hx - p.coord.x); // tangential
    ll.iter(p..m)
        .filter(|p| hx > p.coord.x && p.coord.x >= mp.coord.x)
        .filter(|p| NodeTriangle(n1, mp, n2).contains_point(**p))
        .fold((m, T::max_value() / two), |(m, tan_min), p| {
            if ((calctan(p) < tan_min)
                || (calctan(p) == tan_min && p.coord.x > ll.nodes[m].coord.x))
                && locally_inside(ll, p, &ll.nodes[hole])
            {
                (p.idx, calctan(p))
            } else {
                (m, tan_min)
            }
        })
        .0
}

/* check if two segments cross over each other. note this is different
from pure intersction. only two segments crossing over at some interior
point is considered intersection.

line segment p1-q1 vs line segment p2-q2.

note that if they are collinear, or if the end points touch, or if
one touches the other at one point, it is not considered an intersection.

please note that the other algorithms in this earcut code depend on this
interpretation of the concept of intersection - if this is modified
so that endpoint touching qualifies as intersection, then it will have
a problem with certain inputs.

bsed on https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/

this has been modified from the version in earcut.js to remove the
detection for endpoint detection.

    a1=area(p1,q1,p2);a2=area(p1,q1,q2);a3=area(p2,q2,p1);a4=area(p2,q2,q1);
    p1 q1    a1 cw   a2 cw   a3 ccw   a4  ccw  a1==a2  a3==a4  fl
    p2 q2
    p1 p2    a1 ccw  a2 ccw  a3 cw    a4  cw   a1==a2  a3==a4  fl
    q1 q2
    p1 q2    a1 ccw  a2 ccw  a3 ccw   a4  ccw  a1==a2  a3==a4  fl
    q1 p2
    p1 q2    a1 cw   a2 ccw  a3 ccw   a4  cw   a1!=a2  a3!=a4  tr
    p2 q1
*/

fn pseudo_intersects<T: Float>(
    p1: LinkedListNode<T>,
    q1: LinkedListNode<T>,
    p2: LinkedListNode<T>,
    q2: LinkedListNode<T>,
) -> bool {
    if (p1.xy_eq(p2) && q1.xy_eq(q2)) || (p1.xy_eq(q2) && q1.xy_eq(p2)) {
        return true;
    }
    let zero = T::zero();

    (NodeTriangle(p1, q1, p2).area() > zero) != (NodeTriangle(p1, q1, q2).area() > zero)
        && (NodeTriangle(p2, q2, p1).area() > zero) != (NodeTriangle(p2, q2, q1).area() > zero)
}

// check if a polygon diagonal intersects any polygon segments
fn intersects_polygon<T: Float>(
    ll: &LinkedLists<T>,
    a: LinkedListNode<T>,
    b: LinkedListNode<T>,
) -> bool {
    ll.iter_pairs(a.idx..a.idx).any(|(p, n)| {
        p.vertices_index != a.vertices_index
            && n.vertices_index != a.vertices_index
            && p.vertices_index != b.vertices_index
            && n.vertices_index != b.vertices_index
            && pseudo_intersects(*p, *n, a, b)
    })
}

// check if a polygon diagonal is locally inside the polygon
fn locally_inside<T: Float>(
    ll: &LinkedLists<T>,
    a: &LinkedListNode<T>,
    b: &LinkedListNode<T>,
) -> bool {
    let zero = T::zero();

    match NodeTriangle(*prevref!(ll, a.idx), *a, *nextref!(ll, a.idx)).area() < zero {
        true => {
            NodeTriangle(*a, *b, *nextref!(ll, a.idx)).area() >= zero
                && NodeTriangle(*a, *prevref!(ll, a.idx), *b).area() >= zero
        }
        false => {
            NodeTriangle(*a, *b, *prevref!(ll, a.idx)).area() < zero
                || NodeTriangle(*a, *nextref!(ll, a.idx), *b).area() < zero
        }
    }
}

// check if the middle point of a polygon diagonal is inside the polygon
fn middle_inside<T: Float>(
    ll: &LinkedLists<T>,
    a: &LinkedListNode<T>,
    b: &LinkedListNode<T>,
) -> bool {
    let two = T::one() + T::one();

    let (mx, my) = ((a.coord.x + b.coord.x) / two, (a.coord.y + b.coord.y) / two);
    ll.iter_pairs(a.idx..a.idx)
        .filter(|(p, n)| (p.coord.y > my) != (n.coord.y > my))
        .filter(|(p, n)| n.coord.y != p.coord.y)
        .filter(|(p, n)| {
            (mx) < ((n.coord.x - p.coord.x) * (my - p.coord.y) / (n.coord.y - p.coord.y)
                + p.coord.x)
        })
        .fold(false, |inside, _| !inside)
}

/* link two polygon vertices with a bridge;

if the vertices belong to the same linked list, this splits the list
into two new lists, representing two new polygons.

if the vertices belong to separate linked lists, it merges them into a
single linked list.

For example imagine 6 points, labeled with numbers 0 thru 5, in a single cycle.
Now split at points 1 and 4. The 2 new polygon cycles will be like this:
0 1 4 5 0 1 ...  and  1 2 3 4 1 2 3 .... However because we are using linked
lists of nodes, there will be two new nodes, copies of points 1 and 4. So:
the new cycles will be through nodes 0 1 4 5 0 1 ... and 2 3 6 7 2 3 6 7 .

splitting algorithm:

.0...1...2...3...4...5...     6     7
5p1 0a2 1m3 2n4 3b5 4q0      .c.   .d.

an<-2     an = a.next,
bp<-3     bp = b.prev;
1.n<-4    a.next = b;
4.p<-1    b.prev = a;
6.n<-2    c.next = an;
2.p<-6    an.prev = c;
7.n<-6    d.next = c;
6.p<-7    c.prev = d;
3.n<-7    bp.next = d;
7.p<-3    d.prev = bp;

result of split:
<0...1> <2...3> <4...5>      <6....7>
5p1 0a4 6m3 2n7 1b5 4q0      7c2  3d6
      x x     x x            x x  x x    // x shows links changed

a b q p a b q p  // begin at a, go next (new cycle 1)
a p q b a p q b  // begin at a, go prev (new cycle 1)
m n d c m n d c  // begin at m, go next (new cycle 2)
m c d n m c d n  // begin at m, go prev (new cycle 2)

Now imagine that we have two cycles, and
they are 0 1 2, and 3 4 5. Split at points 1 and
4 will result in a single, long cycle,
0 1 4 5 3 7 6 2 0 1 4 5 ..., where 6 and 1 have the
same x y f64s, as do 7 and 4.

 0...1...2   3...4...5        6     7
2p1 0a2 1m0 5n4 3b5 4q3      .c.   .d.

an<-2     an = a.next,
bp<-3     bp = b.prev;
1.n<-4    a.next = b;
4.p<-1    b.prev = a;
6.n<-2    c.next = an;
2.p<-6    an.prev = c;
7.n<-6    d.next = c;
6.p<-7    c.prev = d;
3.n<-7    bp.next = d;
7.p<-3    d.prev = bp;

result of split:
 0...1...2   3...4...5        6.....7
2p1 0a4 6m0 5n7 1b5 4q3      7c2   3d6
      x x     x x            x x   x x

a b q n d c m p a b q n d c m .. // begin at a, go next
a p m c d n q b a p m c d n q .. // begin at a, go prev

Return value.

Return value is the new node, at point 7.
*/
fn split_bridge_polygon<T: Float>(
    ll: &mut LinkedLists<T>,
    a: LinkedListNodeIndex,
    b: LinkedListNodeIndex,
) -> LinkedListNodeIndex {
    let cidx = ll.nodes.len();
    let didx = cidx + 1;
    let mut c = LinkedListNode::new(ll.nodes[a].vertices_index, ll.nodes[a].coord, cidx);
    let mut d = LinkedListNode::new(ll.nodes[b].vertices_index, ll.nodes[b].coord, didx);

    let an = ll.nodes[a].next_linked_list_node_index;
    let bp = ll.nodes[b].prev_linked_list_node_index;

    ll.nodes[a].next_linked_list_node_index = b;
    ll.nodes[b].prev_linked_list_node_index = a;

    c.next_linked_list_node_index = an;
    ll.nodes[an].prev_linked_list_node_index = cidx;

    d.next_linked_list_node_index = cidx;
    c.prev_linked_list_node_index = didx;

    ll.nodes[bp].next_linked_list_node_index = didx;
    d.prev_linked_list_node_index = bp;

    ll.nodes.push(c);
    ll.nodes.push(d);
    didx
}
