extern crate earcutr;
extern crate serde;
extern crate serde_json;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::Read;
use std::io::Write;
use std::time::Instant;

fn mkoutput(filename_w_dashes: &str, triangles: Vec<usize>) {
    let filename = str::replace(filename_w_dashes, "-", "_");
    let outfile = &format!("examples/output/{}.js", filename);
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
                        for points in contour {
                            let points = points.as_array().unwrap();
                            let mut vp: Vec<f64> = Vec::new();
                            for val in points {
                                let val = val.to_string();
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

fn benchy(nm: &str) {
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    let now = Instant::now();
    let iters = 99u32;

    println!("report for {}", nm);
    for _ in 0..iters {
        triangles = earcutr::earcut(&data, &holeidxs, dimensions).unwrap();
    }
    println!("{:?}", triangles);
    println!("num tris {}", triangles.len() / 3);
    println!(
        "Duration: {} seconds and {} nanoseconds",
        now.elapsed().as_secs(),
        now.elapsed().subsec_nanos()
    );
    println!("ns/iter {}", now.elapsed().subsec_nanos() / (iters));

    mkoutput(nm, triangles);
}

fn main() {
    benchy("water");
    //	benchy("water2");
}
