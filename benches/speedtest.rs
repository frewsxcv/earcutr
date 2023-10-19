use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::fs::File;
use std::fs::OpenOptions;
use std::io::Read;
use std::io::Write;

// this is to "force" optimized code to measure results, by outputting
fn mkoutput(filename_w_dashes: &str, triangles: &[usize]) {
    let filename = str::replace(filename_w_dashes, "-", "_");
    let outfile = &format!("benches/benchoutput/{}.js", filename);
    match OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(outfile)
    {
        Err(e) => println!("error writing {} {}", outfile, e),
        Ok(f) => writeln!(
            &f,
            r###"testOutput["{}"]["benchmark"]=[{:?},{:?},{:?}];"###,
            filename,
            0,
            triangles.len(),
            triangles
        )
        .unwrap(),
    };
}

fn parse_json(rawdata: &str) -> Option<Vec<Vec<Vec<f64>>>> {
    let mut v: Vec<Vec<Vec<f64>>> = Vec::new();
    match serde_json::from_str::<serde_json::Value>(rawdata) {
        Err(e) => println!("error deserializing, {}", e),
        Ok(jsondata) => {
            if jsondata.is_array() {
                let contours = jsondata.as_array().unwrap();
                for contourval in contours {
                    if contourval.is_array() {
                        let contour = contourval.as_array().unwrap();
                        let mut vc: Vec<Vec<f64>> = Vec::new();
                        for j in contour {
                            let points = j.as_array().unwrap();
                            let mut vp: Vec<f64> = Vec::new();
                            for k in points {
                                let val = k.to_string();
                                let pval = val.parse::<f64>().unwrap();
                                vp.push(pval);
                            }
                            vc.push(vp);
                        }
                        v.push(vc);
                    }
                }
            }
        }
    };
    Some(v)
}

fn load_json(testname: &str) -> (Vec<f64>, Vec<usize>, usize) {
    let fullname = format!("./tests/fixtures/{}.json", testname);
    let mut xdata: Vec<Vec<Vec<f64>>> = Vec::new();
    match File::open(&fullname) {
        Err(why) => println!("failed to open file '{}': {}", fullname, why),
        Ok(mut f) => {
            //println!("testing {},", fullname);
            let mut strdata = String::new();
            match f.read_to_string(&mut strdata) {
                Err(why) => println!("failed to read {}, {}", fullname, why),
                Ok(_numb) => {
                    //println!("read {} bytes", numb);
                    let rawstring = strdata.trim();
                    match parse_json(rawstring) {
                        None => println!("failed to parse {}", fullname),
                        Some(parsed_data) => {
                            xdata = parsed_data;
                        }
                    };
                }
            };
        }
    };
    earcutr::legacy::flatten(&xdata)
}

fn bench_quadrilateral(criterion: &mut Criterion) {
    criterion.bench_function("bench_quadrilateral", |bench| {
        bench.iter(|| {
            black_box(earcutr::earcut(
                &[10., 0., 0., 50., 60., 60., 70., 10.],
                &[],
                2,
            ));
        });
    });
}

fn bench_hole(criterion: &mut Criterion) {
    let mut v = vec![0., 0., 50., 0., 50., 50., 0., 50.];
    let h = vec![10., 10., 40., 10., 40., 40., 10., 40.];
    v.extend(h);
    criterion.bench_function("bench_hole", |bench| {
        bench.iter(|| {
            black_box(earcutr::earcut(&v, &[4], 2));
        })
    });
}

fn bench_flatten(criterion: &mut Criterion) {
    let v = vec![
        vec![vec![0., 0.], vec![1., 0.], vec![1., 1.], vec![0., 1.]], // outer ring
        vec![vec![1., 1.], vec![3., 1.], vec![3., 3.]],               // hole ring
    ];
    criterion.bench_function("bench_flatten", |bench| {
        bench.iter(|| {
            let (_vertices, _holes, _dimensions) = black_box(earcutr::legacy::flatten(&v));
        })
    });
}

fn bench_indices_2d(criterion: &mut Criterion) {
    criterion.bench_function("bench_indices_2d", |bench| {
        bench.iter(|| {
            let _indices = black_box(earcutr::earcut(
                &[10.0, 0.0, 0.0, 50.0, 60.0, 60.0, 70.0, 10.0],
                &[],
                2,
            ));
        })
    });
}

fn bench_indices_3d(criterion: &mut Criterion) {
    criterion.bench_function("bench_indices_3d", |bench| {
        bench.iter(|| {
            let _indices = black_box(earcutr::earcut(
                &[
                    10.0, 0.0, 0.0, 0.0, 50.0, 0.0, 60.0, 60.0, 0.0, 70.0, 10.0, 0.0,
                ],
                &[],
                3,
            ));
        })
    });
}

fn bench_empty(criterion: &mut Criterion) {
    criterion.bench_function("bench_empty", |bench| {
        bench.iter(|| {
            let _indices = black_box(earcutr::earcut::<f32>(&[], &[], 2));
        })
    });
}

// file based tests

fn bench_building(criterion: &mut Criterion) {
    let nm = "building";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    criterion.bench_function("bench_building", |bench| {
        bench.iter(|| {
            triangles = black_box(earcutr::earcut(&data, &holeidxs, dimensions).unwrap());
        });
        mkoutput(nm, &triangles);
    });
}

fn bench_dude(criterion: &mut Criterion) {
    let nm = "dude";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    criterion.bench_function("bench_dude", |bench| {
        bench.iter(|| {
            triangles = black_box(earcutr::earcut(&data, &holeidxs, dimensions).unwrap());
        });
        mkoutput(nm, &triangles);
    });
}

fn bench_water(criterion: &mut Criterion) {
    let nm = "water";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    criterion.bench_function("bench_water", |bench| {
        bench.iter(|| {
            triangles = black_box(earcutr::earcut(&data, &holeidxs, dimensions).unwrap());
        });
        mkoutput(nm, &triangles);
    });
}

fn bench_water2(criterion: &mut Criterion) {
    let nm = "water2";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    criterion.bench_function("bench_water2", |bench| {
        bench.iter(|| {
            triangles = black_box(earcutr::earcut(&data, &holeidxs, dimensions).unwrap());
        });
        mkoutput(nm, &triangles);
    });
}

fn bench_water3(criterion: &mut Criterion) {
    let nm = "water3";

    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    criterion.bench_function("bench_water3", |bench| {
        bench.iter(|| {
            triangles = black_box(earcutr::earcut(&data, &holeidxs, dimensions).unwrap());
        });
        mkoutput(nm, &triangles);
    });
}

fn bench_water3b(criterion: &mut Criterion) {
    let nm = "water3b";

    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    criterion.bench_function("bench_water3b", |bench| {
        bench.iter(|| {
            triangles = black_box(earcutr::earcut(&data, &holeidxs, dimensions).unwrap());
        });
        mkoutput(nm, &triangles);
    });
}

fn bench_water4(criterion: &mut Criterion) {
    let nm = "water4";

    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    criterion.bench_function("bench_water4", |bench| {
        bench.iter(|| {
            triangles = black_box(earcutr::earcut(&data, &holeidxs, dimensions).unwrap());
        });
        mkoutput(nm, &triangles);
    });
}

fn bench_water_huge(criterion: &mut Criterion) {
    let nm = "water-huge";

    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    //PROFILER.lock().unwrap().start("./earcutr.profile").unwrap();
    criterion.bench_function("bench_water_huge", |bench| {
        bench.iter(|| {
            //	for i in 0..99 {
            triangles = black_box(earcutr::earcut(&data, &holeidxs, dimensions).unwrap());
            //	}
        });
        mkoutput(nm, &triangles);
    });
    //PROFILER.lock().unwrap().stop().unwrap();
}

fn bench_water_huge2(criterion: &mut Criterion) {
    let nm = "water-huge2";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    criterion.bench_function("bench_water_huge2", |bench| {
        bench.iter(|| {
            triangles = black_box(earcutr::earcut(&data, &holeidxs, dimensions).unwrap());
        });
        mkoutput(nm, &triangles);
    });
}

fn bench_degenerate(criterion: &mut Criterion) {
    let nm = "degenerate";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    criterion.bench_function("bench_degenerate", |bench| {
        bench.iter(|| {
            triangles = black_box(earcutr::earcut(&data, &holeidxs, dimensions).unwrap());
        });
        mkoutput(nm, &triangles);
    });
}

fn bench_bad_hole(criterion: &mut Criterion) {
    let nm = "bad-hole";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    criterion.bench_function("bench_bad_hole", |bench| {
        bench.iter(|| {
            triangles = black_box(earcutr::earcut(&data, &holeidxs, dimensions).unwrap());
        });
        mkoutput(nm, &triangles);
    });
}

fn bench_empty_square(criterion: &mut Criterion) {
    let nm = "empty-square";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    criterion.bench_function("bench_empty_square", |bench| {
        bench.iter(|| {
            triangles = black_box(earcutr::earcut(&data, &holeidxs, dimensions).unwrap());
        });
        mkoutput(nm, &triangles);
    });
}

fn bench_issue16(criterion: &mut Criterion) {
    let nm = "issue16";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    criterion.bench_function("bench_issue16", |bench| {
        bench.iter(|| {
            triangles = black_box(earcutr::earcut(&data, &holeidxs, dimensions).unwrap());
        });
        mkoutput(nm, &triangles);
    });
}

fn bench_issue17(criterion: &mut Criterion) {
    let nm = "issue17";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    criterion.bench_function("bench_issue17", |bench| {
        bench.iter(|| {
            triangles = black_box(earcutr::earcut(&data, &holeidxs, dimensions).unwrap());
        });
        mkoutput(nm, &triangles);
    });
}

fn bench_steiner(criterion: &mut Criterion) {
    let nm = "steiner";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    criterion.bench_function("bench_steiner", |bench| {
        bench.iter(|| {
            triangles = black_box(earcutr::earcut(&data, &holeidxs, dimensions).unwrap());
        });
        mkoutput(nm, &triangles);
    });
}

fn bench_issue29(criterion: &mut Criterion) {
    let nm = "issue29";
    let (data, holeidxs, dimensions) = load_json(nm);

    let mut triangles = Vec::new();
    criterion.bench_function("bench_issue29", |bench| {
        bench.iter(|| {
            triangles = black_box(earcutr::earcut(&data, &holeidxs, dimensions).unwrap());
        });
        mkoutput(nm, &triangles);
    });
}

fn bench_issue34(criterion: &mut Criterion) {
    let nm = "issue34";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    criterion.bench_function("bench_issue34", |bench| {
        bench.iter(|| {
            triangles = black_box(earcutr::earcut(&data, &holeidxs, dimensions).unwrap());
        });
        mkoutput(nm, &triangles);
    });
}

fn bench_issue35(criterion: &mut Criterion) {
    let nm = "issue35";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    criterion.bench_function("bench_issue35", |bench| {
        bench.iter(|| {
            triangles = black_box(earcutr::earcut(&data, &holeidxs, dimensions).unwrap());
        });
        mkoutput(nm, &triangles);
    });
}

fn bench_self_touching(criterion: &mut Criterion) {
    let nm = "self-touching";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    criterion.bench_function("bench_self_touching", |bench| {
        bench.iter(|| {
            triangles = black_box(earcutr::earcut(&data, &holeidxs, dimensions).unwrap());
        });
        mkoutput(nm, &triangles);
    });
}

fn bench_outside_ring(criterion: &mut Criterion) {
    let nm = "outside-ring";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    criterion.bench_function("bench_outside_ring", |bench| {
        bench.iter(|| {
            triangles = black_box(earcutr::earcut(&data, &holeidxs, dimensions).unwrap());
        });
        mkoutput(nm, &triangles);
    });
}

fn bench_simplified_us_border(criterion: &mut Criterion) {
    let nm = "simplified-us-border";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    criterion.bench_function("bench_simplified_us_border", |bench| {
        bench.iter(|| {
            triangles = black_box(earcutr::earcut(&data, &holeidxs, dimensions).unwrap());
        });
        mkoutput(nm, &triangles);
    });
}

fn bench_touching_holes(criterion: &mut Criterion) {
    let nm = "touching-holes";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    criterion.bench_function("bench_touching_holes", |bench| {
        bench.iter(|| {
            triangles = black_box(earcutr::earcut(&data, &holeidxs, dimensions).unwrap());
        });
        mkoutput(nm, &triangles);
    });
}

fn bench_hole_touching_outer(criterion: &mut Criterion) {
    let nm = "hole-touching-outer";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    criterion.bench_function("bench_hole_touching_outer", |bench| {
        bench.iter(|| {
            triangles = black_box(earcutr::earcut(&data, &holeidxs, dimensions).unwrap());
        });
        mkoutput(nm, &triangles);
    });
}

fn bench_hilbert(criterion: &mut Criterion) {
    let nm = "hilbert";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    criterion.bench_function("bench_hilbert", |bench| {
        bench.iter(|| {
            triangles = black_box(earcutr::earcut(&data, &holeidxs, dimensions).unwrap());
        });
        mkoutput(nm, &triangles);
    });
}

fn bench_issue45(criterion: &mut Criterion) {
    let nm = "issue45";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    criterion.bench_function("bench_issue45", |bench| {
        bench.iter(|| {
            triangles = black_box(earcutr::earcut(&data, &holeidxs, dimensions).unwrap());
        });
        mkoutput(nm, &triangles);
    });
}

fn bench_eberly_3(criterion: &mut Criterion) {
    let nm = "eberly-3";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    criterion.bench_function("bench_eberly_3", |bench| {
        bench.iter(|| {
            triangles = black_box(earcutr::earcut(&data, &holeidxs, dimensions).unwrap());
        });
        mkoutput(nm, &triangles);
    });
}

fn bench_eberly_6(criterion: &mut Criterion) {
    let nm = "eberly-6";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    criterion.bench_function("bench_eberly_6", |bench| {
        bench.iter(|| {
            triangles = black_box(earcutr::earcut(&data, &holeidxs, dimensions).unwrap());
        });
        mkoutput(nm, &triangles);
    });
}

fn bench_issue52(criterion: &mut Criterion) {
    let nm = "issue52";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    criterion.bench_function("bench_issue52", |bench| {
        bench.iter(|| {
            triangles = black_box(earcutr::earcut(&data, &holeidxs, dimensions).unwrap());
        });
        mkoutput(nm, &triangles);
    });
}

fn bench_shared_points(criterion: &mut Criterion) {
    let nm = "shared-points";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    criterion.bench_function("bench_shared_points", |bench| {
        bench.iter(|| {
            triangles = black_box(earcutr::earcut(&data, &holeidxs, dimensions).unwrap());
        });
        mkoutput(nm, &triangles);
    });
}

fn bench_bad_diagonals(criterion: &mut Criterion) {
    let nm = "bad-diagonals";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    criterion.bench_function("bench_bad_diagonals", |bench| {
        bench.iter(|| {
            triangles = black_box(earcutr::earcut(&data, &holeidxs, dimensions).unwrap());
        });
        mkoutput(nm, &triangles);
    });
}

fn bench_issue83(criterion: &mut Criterion) {
    let nm = "issue83";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    criterion.bench_function("bench_issue83", |bench| {
        bench.iter(|| {
            triangles = black_box(earcutr::earcut(&data, &holeidxs, dimensions).unwrap());
        });
        mkoutput(nm, &triangles);
    });
}

criterion_group!(
    benches,
    bench_indices_3d,
    bench_indices_2d,
    bench_empty,
    bench_quadrilateral,
    bench_hole,
    bench_flatten,
    bench_bad_diagonals,
    bench_bad_hole,
    bench_building,
    bench_degenerate,
    bench_dude,
    bench_eberly_3,
    bench_eberly_6,
    bench_empty_square,
    bench_hilbert,
    bench_hole_touching_outer,
    bench_issue16,
    bench_issue17,
    bench_issue29,
    bench_issue34,
    bench_issue35,
    bench_issue45,
    bench_issue52,
    bench_issue83,
    bench_outside_ring,
    bench_self_touching,
    bench_shared_points,
    bench_simplified_us_border,
    bench_steiner,
    bench_touching_holes,
    bench_water_huge,
    bench_water_huge2,
    bench_water,
    bench_water2,
    bench_water3,
    bench_water3b,
    bench_water4,
);
criterion_main!(benches);
