use std::io::Write;

pub fn write_vengi(
    filename: &str,
    slice: &[u8],
    w: usize,
    h: usize,
    d: usize,
) -> std::io::Result<()> {
    let mut file = std::fs::File::create(filename)?;

    file.write("VENG".as_bytes())?;

    let mut file = flate2::write::ZlibEncoder::new(file, flate2::Compression::fast());
    //let mut file = zstd::stream::write::Encoder::new(file, 0)?.auto_finish();

    file.write(&4_u32.to_le_bytes())?;

    let mut indices = [0; 256];
    for (i, v) in indices.iter_mut().enumerate() {
        *v = i as u8;
    }

    let palc = Palc {
        color_abgr: &[[0, 0, 0, 255]; 256],
        emit_color_abgr: &[[0, 0, 0, 0]; 256],
        indices: &indices,
    };

    let anim = Anim {
        name: "Default",
        keyframes: &[Keyframe {
            frame_index: 0,
            long_rotation: false,
            interpolation_type: "Linear",
            local_matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }],
    };

    write_node(
        &Node {
            name: "root",
            ty: "Root",
            id: 0,
            reference_id: -1,
            visibility: true,
            lock_state: false,
            color_abgr: [0; 4],
            pivot: [0.0; 3],
            palc: palc.clone(),
            anim: anim.clone(),
            data: None,
            children: &[Node {
                name: "node 1",
                ty: "Model",
                id: 0,
                reference_id: -1,
                visibility: true,
                lock_state: false,
                color_abgr: [0; 4],
                pivot: [0.0; 3],
                palc,
                anim,
                data: Some(Data {
                    region: [0, 0, 0, (w - 1) as _, (h - 1) as _, (d - 1) as _],
                    dense: slice,
                }),
                children: &[],
            }],
        },
        &mut file,
    )?;

    Ok(())
}

struct Keyframe<'a> {
    frame_index: u32,
    long_rotation: bool,
    interpolation_type: &'a str,
    local_matrix: [[f32; 4]; 4],
}

#[derive(Clone)]
struct Anim<'a> {
    name: &'a str,
    keyframes: &'a [Keyframe<'a>],
}

fn write_keyframe<W: std::io::Write>(keyframe: &Keyframe, file: &mut W) -> std::io::Result<()> {
    file.write("KEYF".as_bytes())?;
    file.write(&keyframe.frame_index.to_le_bytes())?;
    file.write(&[keyframe.long_rotation as u8])?;
    file.write(&(keyframe.interpolation_type.len() as u16).to_le_bytes())?;
    file.write(&keyframe.interpolation_type.as_bytes())?;
    file.write(bytemuck::cast_slice(&keyframe.local_matrix))?;
    Ok(())
}

fn write_anim<W: std::io::Write>(anim: &Anim, file: &mut W) -> std::io::Result<()> {
    file.write("ANIM".as_bytes())?;
    file.write(&(anim.name.len() as u16).to_le_bytes())?;
    file.write(&anim.name.as_bytes())?;
    for keyframe in anim.keyframes {
        write_keyframe(keyframe, file)?;
    }
    file.write("ENDA".as_bytes())?;
    Ok(())
}

#[derive(Clone)]
struct Palc<'a> {
    color_abgr: &'a [[u8; 4]],
    emit_color_abgr: &'a [[u8; 4]],
    indices: &'a [u8],
}

fn write_palc<W: std::io::Write>(palc: &Palc, file: &mut W) -> std::io::Result<()> {
    file.write("PALC".as_bytes())?;
    file.write(&(palc.color_abgr.len() as u32).to_le_bytes())?;
    file.write(bytemuck::cast_slice(&palc.color_abgr))?;
    file.write(bytemuck::cast_slice(&palc.emit_color_abgr))?;
    file.write(bytemuck::cast_slice(&palc.indices))?;
    file.write(&0_u32.to_le_bytes())?;

    Ok(())
}

struct Data<'a> {
    region: [i32; 6],
    dense: &'a [u8],
}

struct Node<'a> {
    name: &'a str,
    ty: &'a str,
    id: i32,
    reference_id: i32,
    visibility: bool,
    lock_state: bool,
    color_abgr: [u8; 4],
    pivot: [f32; 3],
    palc: Palc<'a>,
    anim: Anim<'a>,
    data: Option<Data<'a>>,
    children: &'a [Node<'a>],
}

fn write_data<W: std::io::Write>(data: &Data, file: &mut W) -> std::io::Result<()> {
    file.write("DATA".as_bytes())?;
    file.write(bytemuck::cast_slice(&data.region))?;

    for &byte in data.dense {
        let air = byte == 0;
        file.write(&[air as u8])?;
        if !air {
            file.write(&[byte, 0])?;
        }
    }

    Ok(())
}

fn write_node<W: std::io::Write>(node: &Node, file: &mut W) -> std::io::Result<()> {
    file.write("NODE".as_bytes())?;
    file.write(&(node.name.len() as u16).to_le_bytes())?;
    file.write(&node.name.as_bytes())?;
    file.write(&(node.ty.len() as u16).to_le_bytes())?;
    file.write(node.ty.as_bytes())?;
    file.write(&node.id.to_le_bytes())?;
    file.write(&node.reference_id.to_le_bytes())?;
    file.write(&[node.visibility as u8, node.lock_state as u8])?;
    file.write(&node.color_abgr)?;
    file.write(bytemuck::cast_slice(&node.pivot))?;
    write_palc(&node.palc, file)?;
    if let Some(data) = node.data.as_ref() {
        write_data(data, file)?;
    }

    write_anim(&node.anim, file)?;
    for child in node.children {
        write_node(child, file)?;
    }
    file.write("ENDN".as_bytes())?;
    Ok(())
}
