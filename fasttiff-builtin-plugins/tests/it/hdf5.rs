#![cfg(all(feature = "plugin-hdf5", not(target_arch = "wasm32")))]
use fast_tiff_viewer::plugins::Registry;
use fasttiff_plugin_api::{
    Confidence, ImportHost, ImportRequest, Importer, ParamKind, ParamValue, Params, PixelType,
    PlaneData,
};
use std::path::Path;

struct Host;
impl ImportHost for Host {
    fn progress(&mut self, _: f32) -> bool {
        true
    }
    fn log(&mut self, _: &str) {}
}

#[test]
fn select_one_dataset_through_the_registry() {
    // Small local fixture with a 512-byte HDF5 user block, a cyclic group link,
    // a coordinate vector, a deflated signed volume, RGB pixels and a two-page TIFF.
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/hdf5-selection.h5");
    let mut reg = Registry::new(fasttiff_builtin_plugins::register);
    let head = Registry::read_head(&path);
    let (index, confidence) = reg.importers_for(&path, &head)[0];
    assert_eq!(confidence, Confidence::Certain);
    let plugin = &mut reg.importer_mut(index).unwrap().importer;
    let decls = plugin.dialog(&path).unwrap();
    let [choice] = &decls[..] else {
        panic!("expected dataset picker: {decls:?}")
    };
    let ParamKind::Choice { options, .. } = &choice.kind else {
        panic!("expected choices")
    };
    assert_eq!(options.len(), 4);
    let mut read = |dataset: &str| {
        let i = options.iter().position(|s| s.starts_with(dataset)).unwrap();
        let mut params = Params::new();
        params.set(choice.key.clone(), ParamValue::Choice(i));
        let result = plugin
            .import(
                &ImportRequest {
                    path: path.clone(),
                    params,
                },
                &mut Host,
            )
            .unwrap();
        result.image.validate().unwrap();
        assert!(result.info.unwrap().description.unwrap().contains(dataset));
        result.image
    };
    let volume = read("/images/stack");
    assert_eq!(
        (
            volume.width,
            volume.height,
            volume.slices,
            volume.pixel_type
        ),
        (5, 4, 3, PixelType::I16)
    );
    assert_eq!(
        volume.planes[0],
        PlaneData::U16((-20i16..0).map(|v| v as u16).collect())
    );
    assert_eq!(volume.planes[2], PlaneData::U16((20..40).collect()));
    let rgb = read("/images/color");
    assert_eq!((rgb.width, rgb.height, rgb.channels), (3, 2, 3));
    assert_eq!(rgb.planes[1], PlaneData::U8(vec![1, 4, 7, 10, 13, 16]));
    let xy = read("/images/xy");
    assert_eq!((xy.width, xy.height), (3, 2));
    assert_eq!(xy.planes[0], PlaneData::U8(vec![0, 2, 4, 1, 3, 5]));
    let tif = read("/images/bytes");
    assert_eq!((tif.width, tif.height, tif.planes.len()), (3, 2, 2));
    assert_eq!(tif.planes[1], PlaneData::U16((6..12).collect()));
    assert!(plugin
        .import(
            &ImportRequest {
                path,
                params: Params::new()
            },
            &mut Host
        )
        .is_err());
}

#[test]
fn matching_images_become_frames_in_dataset_path_order() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/hdf5-groups.h5");
    let mut reg = Registry::new(fasttiff_builtin_plugins::register);
    let (index, _) = reg.importers_for(&path, &Registry::read_head(&path))[0];
    let plugin = &mut reg.importer_mut(index).unwrap().importer;
    let decls = plugin.dialog(&path).unwrap();
    let ParamKind::Choice { options, .. } = &decls[0].kind else {
        panic!("expected dataset picker")
    };
    assert_eq!(
        &options[..2],
        ["All 30x30 images (5 frames)", "All 80x80 images (3 frames)"]
    );
    assert_eq!(options.len(), 2 + 11);
    assert!(options[2..].iter().all(|label| label.starts_with('/')));
    let mut read = |index| {
        let mut params = Params::new();
        params.set(&decls[0].key, ParamValue::Choice(index));
        plugin
            .import(
                &ImportRequest {
                    path: path.clone(),
                    params,
                },
                &mut Host,
            )
            .unwrap()
    };
    let result = read(0);
    let image = result.image;
    image.validate().unwrap();
    assert_eq!(
        (
            image.width,
            image.height,
            image.frames,
            image.slices,
            image.channels
        ),
        (30, 30, 5, 1, 1)
    );
    for (frame, expected) in image.planes.iter().zip([255., -32768., 65535., 4., 5.]) {
        assert_eq!(frame, &PlaneData::F32(vec![expected; 900]));
    }
    assert!(result
        .info
        .unwrap()
        .description
        .unwrap()
        .contains("/small/1\n/small/2.tif\n/small/3"));
    let image = read(1).image;
    image.validate().unwrap();
    assert_eq!(
        (image.width, image.height, image.frames, image.pixel_type),
        (80, 80, 3, PixelType::U16)
    );
    assert_eq!(image.planes[2], PlaneData::U16(vec![2; 6400]));
    let single = read(2).image;
    assert_eq!((single.width, single.height, single.frames), (80, 80, 1));
}
