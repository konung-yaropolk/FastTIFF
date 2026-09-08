//! The XML → text translation, against a synthetic acquisition.
//!
//! The fixture below is not a real file: it is the *shape* of one, with the
//! same element names, namespaces and nesting, and values chosen so that every
//! mapping is visible in the output and a mis-wiring shows up as a wrong
//! number rather than a missing one. The frame is deliberately not square and
//! its pixels deliberately are not either, because an x/y swap is the mistake
//! this kind of code makes and a square fixture cannot catch it.
//!
//! What this cannot check is that the mapping is *right* — that
//! `commonimage:length/x` really is the pixel size FluoView calls
//! `[um/pixel]`. Only the vendor's own export can settle that, and it did: see
//! the module docs. These tests hold the translation to what that comparison
//! established.

use super::*;

/// One acquisition's metadata documents, in the order an OIR carries them.
fn documents() -> Vec<String> {
    vec![
        // Written when the acquisition starts: the events have not happened.
        r#"<?xml version="1.0" encoding="ASCII"?>
           <event:eventList xmlns:event="http://www.olympus.co.jp/hpf/model/event"/>"#
            .to_string(),
        r#"<?xml version="1.0" encoding="ASCII"?>
           <fileinfo:fileInfomation xmlns:fileinfo="http://www.olympus.co.jp/hpf/model/fileinfo">
             <fileinfo:version>2.1.2.3</fileinfo:version>
           </fileinfo:fileInfomation>"#
            .to_string(),
        properties(),
        frame(0, 0.0),
        frame(1, 1000.0),
        frame(2, 2500.0),
        // Written at the end, and the only copy that knows what happened.
        r#"<?xml version="1.0" encoding="ASCII"?>
           <event:eventList xmlns:event="http://www.olympus.co.jp/hpf/model/event">
             <event:event>
               <event:name>DRS</event:name>
               <event:time>27772.762</event:time>
               <event:type>TTL_OUT</event:type>
             </event:event>
             <event:event>
               <event:name>Drug application</event:name>
               <event:time>75246.282</event:time>
               <event:type>TTL_OUT</event:type>
             </event:event>
           </event:eventList>"#
            .to_string(),
    ]
}

/// One frame's document, carrying its timestamp in milliseconds.
fn frame(index: usize, ms: f64) -> String {
    format!(
        r#"<?xml version="1.0" encoding="ASCII"?>
           <lsmframe:frameProperties
               xmlns:base="http://www.olympus.co.jp/hpf/model/base"
               xmlns:commonframe="http://www.olympus.co.jp/hpf/model/commonframe"
               xmlns:lsmframe="http://www.olympus.co.jp/hpf/model/lsmframe"
               id="t{n:03}_0_1">
             <commonframe:general><base:name>t{n:03}_0_1</base:name></commonframe:general>
             <commonframe:imageDefinition>
               <base:width>512</base:width>
               <base:height>256</base:height>
               <base:depth>2</base:depth>
             </commonframe:imageDefinition>
             <commonframe:axisValue>
               <commonframe:axisType>TIMELAPSE</commonframe:axisType>
               <commonframe:position>{ms:?}</commonframe:position>
             </commonframe:axisValue>
           </lsmframe:frameProperties>"#,
        n = index + 1,
    )
}

/// The acquisition record. Two detectors are configured and both recorded, and
/// the recorded list is in the opposite order to the configured one — so
/// anything matching by position rather than by channel id gets the PMT
/// voltages, detector names and mirrors of the wrong channel.
fn properties() -> String {
    r#"<?xml version="1.0" encoding="ASCII"?>
       <lsmimage:imageProperties
           xmlns:base="http://www.olympus.co.jp/hpf/model/base"
           xmlns:commonimage="http://www.olympus.co.jp/hpf/model/commonimage"
           xmlns:commonparam="http://www.olympus.co.jp/hpf/model/commonparam"
           xmlns:commonphase="http://www.olympus.co.jp/hpf/model/commonphase"
           xmlns:fvCommonphase="http://www.olympus.co.jp/fluoview/model/fv_commonphase"
           xmlns:lsmimage="http://www.olympus.co.jp/hpf/model/lsmimage"
           xmlns:lsmparam="http://www.olympus.co.jp/hpf/model/lsmparam"
           xmlns:opticalelement="http://www.olympus.co.jp/hpf/model/opticalelement"
           version="1.1.0.0">
         <commonimage:general>
           <base:creationDateTime>2025-07-28T15:36:14.925-04:00</base:creationDateTime>
         </commonimage:general>
         <commonimage:system>
           <base:systemName>FVMPE-RS</base:systemName>
           <base:systemVersion>2.3.2.169</base:systemVersion>
         </commonimage:system>
         <commonimage:imageInfo>
           <commonimage:phase><commonimage:group>
             <commonimage:channel id="ch-b" order="2">
               <commonimage:productData><fvCommonphase:filterName>BA575-645</fvCommonphase:filterName></commonimage:productData>
               <commonphase:name>CH2</commonphase:name>
               <commonphase:opticalResolution/>
               <commonphase:length>
                 <commonparam:x>0.621480569402239</commonparam:x>
                 <commonparam:y>0.310740284701119</commonparam:y>
                 <commonparam:z>1.0</commonparam:z>
               </commonphase:length>
               <commonimage:imageDefinition><commonimage:bitCounts>12</commonimage:bitCounts></commonimage:imageDefinition>
             </commonimage:channel>
             <commonimage:channel id="ch-a" order="1">
               <commonimage:productData><fvCommonphase:filterName>BA495-540</fvCommonphase:filterName></commonimage:productData>
               <commonphase:name>CH1</commonphase:name>
               <commonphase:length>
                 <commonparam:x>0.621480569402239</commonparam:x>
                 <commonparam:y>0.310740284701119</commonparam:y>
               </commonphase:length>
               <commonimage:imageDefinition><commonimage:bitCounts>12</commonimage:bitCounts></commonimage:imageDefinition>
             </commonimage:channel>
           </commonimage:group></commonimage:phase>
           <commonimage:axis>
             <commonimage:axis>TIMELAPSE</commonimage:axis>
             <commonimage:step>0.0</commonimage:step>
             <commonimage:maxSize>3</commonimage:maxSize>
           </commonimage:axis>
           <commonimage:width>512</commonimage:width>
           <commonimage:height>256</commonimage:height>
         </commonimage:imageInfo>
         <lsmimage:acquisition>
           <lsmimage:productData><lsmimage:lsmMicroscopePhase><lsmimage:lightpath>
             <opticalelement:firstMirrorCube><opticalelement:name>DM690</opticalelement:name></opticalelement:firstMirrorCube>
           </lsmimage:lightpath></lsmimage:lsmMicroscopePhase></lsmimage:productData>
           <lsmimage:microscopeConfiguration>
             <opticalelement:objectiveLens>
               <opticalelement:name>XLUMPLFLN20XW</opticalelement:name>
               <opticalelement:magnification>20.0</opticalelement:magnification>
               <opticalelement:naValue>1.0</opticalelement:naValue>
             </opticalelement:objectiveLens>
           </lsmimage:microscopeConfiguration>
           <lsmimage:imagingParam>
             <commonparam:pmt channelId="ch-a" detectorId="__D_3__"><lsmparam:voltage>600</lsmparam:voltage></commonparam:pmt>
             <commonparam:pmt channelId="ch-b" detectorId="__D_4__"><lsmparam:voltage>612</lsmparam:voltage></commonparam:pmt>
             <commonparam:mainLaser><commonparam:transmissivity>15.0</commonparam:transmissivity></commonparam:mainLaser>
             <commonparam:method>
               <commonparam:sequentialType>None</commonparam:sequentialType>
               <commonparam:integration><commonparam:type>None</commonparam:type><commonparam:count>5</commonparam:count></commonparam:integration>
             </commonparam:method>
             <commonparam:area>
               <commonparam:xpan>0.0</commonparam:xpan>
               <commonparam:ypan>0.0</commonparam:ypan>
               <commonparam:rotation>0.0</commonparam:rotation>
               <commonparam:zoom>2.0</commonparam:zoom>
             </commonparam:area>
           </lsmimage:imagingParam>
           <lsmimage:phase>
             <commonimage:group>
               <lsmimage:channel id="ch-a" order="1" enable="true" detectorId="__D_3__">
                 <commonphase:name>CH1</commonphase:name>
                 <commonphase:deviceName>RNDD3G</commonphase:deviceName>
                 <commonphase:dyeName>Alexa Fluor 555</commonphase:dyeName>
                 <commonphase:dyeData><commonphase:emissionWavelength>568</commonphase:emissionWavelength></commonphase:dyeData>
               </lsmimage:channel>
               <lsmimage:channel id="ch-b" order="2" enable="true" detectorId="__D_4__">
                 <commonphase:name>CH2</commonphase:name>
                 <commonphase:deviceName>RNDD4G</commonphase:deviceName>
                 <commonphase:dyeName>Alexa Fluor 488</commonphase:dyeName>
                 <commonphase:dyeData><commonphase:emissionWavelength>520</commonphase:emissionWavelength></commonphase:dyeData>
               </lsmimage:channel>
             </commonimage:group>
             <lsmimage:imagingMainLaser><commonparam:wavelength>990</commonparam:wavelength></lsmimage:imagingMainLaser>
           </lsmimage:phase>
           <lsmimage:configuration>
             <lsmimage:productData>
               <lsmimage:externalDetectorLightpath detectorId="__D_3__" detectorIndex="0">
                 <opticalelement:filterCube><opticalelement:dichroicMirror><opticalelement:name>SDM570</opticalelement:name></opticalelement:dichroicMirror></opticalelement:filterCube>
               </lsmimage:externalDetectorLightpath>
               <lsmimage:externalDetectorLightpath detectorId="__D_4__" detectorIndex="1">
                 <opticalelement:filterCube><opticalelement:dichroicMirror><opticalelement:name>SDM505</opticalelement:name></opticalelement:dichroicMirror></opticalelement:filterCube>
               </lsmimage:externalDetectorLightpath>
             </lsmimage:productData>
             <lsmimage:scannerType>Galvano</lsmimage:scannerType>
           </lsmimage:configuration>
           <lsmimage:scannerSettings type="Resonant">
             <lsmimage:param><commonparam:speed><commonparam:speed>0.067</commonparam:speed><commonparam:roundtrip>true</commonparam:roundtrip></commonparam:speed></lsmimage:param>
           </lsmimage:scannerSettings>
           <lsmimage:scannerSettings type="Galvano">
             <lsmimage:param><commonparam:speed><commonparam:speed>2.0</commonparam:speed><commonparam:roundtrip>false</commonparam:roundtrip></commonparam:speed></lsmimage:param>
           </lsmimage:scannerSettings>
         </lsmimage:acquisition>
       </lsmimage:imageProperties>"#
        .to_string()
}

/// The whole translation, as one document. Written out in full rather than
/// probed key by key: the point of this module is that tag 270 gets *this
/// text*, and a golden file is the only way to notice a key that quietly
/// stopped being written.
#[test]
fn the_record_becomes_the_text_the_acquisition_software_exports() {
    let r = Record::parse(&documents());
    let text = r.to_text("Field_1.oir", 3).expect("a record to write");
    let expected = "\
\"[General]\"\t\"\"
\"Name\"\t\"Field_1.oir\"
\"Scan Mode\"\t\"XYT\"
\"Date\"\t\"07/28/2025 03:36:14.925 PM\"
\"File Version\"\t\"2.1.2.3\"
\"System Name\"\t\"FVMPE-RS\"
\"System Version\"\t\"2.3.2.169\"
\"[Dimensions]\"\t\"\"
\"X Dimension\"\t\"512, 0.0 - 318.198 [um], 0.621 [um/pixel]\"
\"Y Dimension\"\t\"256, 0.0 - 79.550 [um], 0.311 [um/pixel]\"
\"Channel Dimension\"\t\"2 [Ch]\"
\"T Dimension\"\t\"3, 0.000 - 2.500 [s], Interval FreeRun\"
\"[Image]\"\t\"\"
\"Image Size\"\t\"512 * 256 [pixel]\"
\"Image Size(Unit Converted)\"\t\"318.198 [um] * 79.550 [um]\"
\"[Acquisition]\"\t\"\"
\"Objective Lens\"\t\"XLUMPLFLN20XW\"
\"Objective Lens Mag.\"\t\"20.0X\"
\"Objective Lens NA\"\t\"1.0\"
\"Scan Device\"\t\"Galvano\"
\"Scan Direction\"\t\"Oneway\"
\"Sampling Speed\"\t\"2.0 [us/pixel]\"
\"Sequential Mode\"\t\"None\"
\"Integration Type\"\t\"None\"
\"Integration Count\"\t\"0\"
\"Rotation\"\t\"0.0 [deg]\"
\"Pan X\"\t\"0.0 [um]\"
\"Pan Y\"\t\"0.0 [um]\"
\"Zoom\"\t\"x2.0\"
\"MirrorTurret 1\"\t\"DM690\"
\"[Channel 1]\"\t\"\"
\"Channel Name\"\t\"RNDD4G\"
\"Dye Name\"\t\"Alexa Fluor 488\"
\"Emission WaveLength\"\t\"520 [nm]\"
\"PMT Voltage\"\t\"612 [V]\"
\"BF Name\"\t\"BA575-645\"
\"Emission DM Name\"\t\"SDM505\"
\"Bits/Pixel\"\t\"12 [bits]\"
\"Laser Wavelength\"\t\"990 [nm]\"
\"Laser Transmissivity\"\t\"15.0 [%]\"
\"[Channel 2]\"\t\"\"
\"Channel Name\"\t\"RNDD3G\"
\"Dye Name\"\t\"Alexa Fluor 555\"
\"Emission WaveLength\"\t\"568 [nm]\"
\"PMT Voltage\"\t\"600 [V]\"
\"BF Name\"\t\"BA495-540\"
\"Emission DM Name\"\t\"SDM570\"
\"Bits/Pixel\"\t\"12 [bits]\"
\"Laser Wavelength\"\t\"990 [nm]\"
\"Laser Transmissivity\"\t\"15.0 [%]\"
\"[Event 1]\"\t\"\"
\"Event Contents\"\t\"DRS\"
\"Event Timer\"\t\"27772.762000[ms]\"
\"[Event 2]\"\t\"\"
\"Event Contents\"\t\"Drug application\"
\"Event Timer\"\t\"75246.282000[ms]\"
";
    // Line by line, so a failure names the line that changed rather than
    // printing two fifty-line blobs.
    for (i, (want, got)) in expected.lines().zip(text.lines()).enumerate() {
        assert_eq!(want, got, "line {}", i + 1);
    }
    assert_eq!(expected.lines().count(), text.lines().count());
    assert_eq!(expected, text);
}

/// Every channel fact — the detector name, the PMT voltage, the emission
/// mirror — is looked up by channel id. The fixture lists the recorded
/// channels in the opposite order to the configured ones, so anything reading
/// them positionally reports the wrong detector's settings under a plausible
/// name, which is the kind of error nobody catches by looking at the picture.
#[test]
fn channel_facts_are_matched_by_id_rather_than_by_position() {
    let r = Record::parse(&documents());
    assert_eq!(r.channels.len(), 2);
    assert_eq!(r.channels[0].label().as_deref(), Some("RNDD4G"));
    assert_eq!(r.channels[0].voltage.as_deref(), Some("612"));
    assert_eq!(r.channels[1].label().as_deref(), Some("RNDD3G"));
    assert_eq!(r.channels[1].voltage.as_deref(), Some("600"));
}

/// An OIR is written as it is acquired, so it holds an event list from before
/// anything happened as well as the real one. Reading the first would produce
/// a recording with no stimulus markers — which does not look like a failure,
/// it looks like an experiment where nothing was triggered.
#[test]
fn the_event_list_written_at_the_end_wins_over_the_empty_one() {
    let r = Record::parse(&documents());
    assert_eq!(r.events.len(), 2, "the empty first list should not win");
    assert_eq!(r.events[0].0, "DRS");
    assert_eq!(r.events[0].1, 27772.762);

    // And the empty one does not erase a list already found, whichever order
    // the documents arrive in.
    let mut reversed = documents();
    reversed.reverse();
    assert_eq!(Record::parse(&reversed).events.len(), 2);
}

/// The scanner that ran, not whichever settings block comes first. A resonant
/// scanner's pixel dwell is thirty times a galvanometer's, so picking the
/// wrong block states a sampling speed the acquisition never used.
#[test]
fn the_settings_of_the_scanner_actually_used_are_the_ones_read() {
    let text = Record::parse(&documents())
        .to_text("f.oir", 3)
        .expect("a record");
    assert!(
        text.contains("\"Sampling Speed\"\t\"2.0 [us/pixel]\""),
        "{text}"
    );
    assert!(
        !text.contains("0.067"),
        "the resonant scanner's speed was read"
    );
}

/// A duration is only stated when the file timed every frame being asked about.
///
/// The timestamps can run out long before the frames do — a split acquisition
/// puts a quarter of them in each of its four files, so reading one file times
/// one quarter of the recording. The last of *those* is the length of a part,
/// and reporting it as the recording's length would put the frame rate out by
/// a factor of four: silently, and in a number every stimulus-locked analysis
/// downstream divides by.
#[test]
fn a_duration_is_stated_only_when_every_frame_was_timed() {
    let mut docs = documents();
    docs.retain(|d| !d.contains("frameProperties") || d.contains("t001"));
    let r = Record::parse(&docs);
    // The first frame is at zero, which is a timestamp and not a duration.
    assert_eq!(r.duration_s(1), None);
    assert_eq!(
        r.duration_s(3),
        None,
        "one timestamp cannot time three frames"
    );

    let text = r.to_text("f.oir", 3).expect("a record");
    // The count is still worth writing; the range is not invented.
    assert!(text.contains("\"T Dimension\"\t\"3 [T]\""), "{text}");
    assert!(!text.contains("0.000 -"), "a range was invented: {text}");
}

/// A split acquisition is several files, and each holds the timestamps of the
/// frames it holds. The recording's timing is in none of them on its own.
#[test]
fn the_timestamps_of_every_part_add_up_to_the_recording() {
    // Part one: the acquisition record and the first two frames. Part two: two
    // more frames, and nothing else — which is what a continuation looks like.
    let all = documents();
    let mut part1 = Reader::default();
    part1.absorb(&all);
    let one = part1.finish();
    assert_eq!(one.duration_s(3), Some(2.5));
    assert_eq!(
        one.duration_s(5),
        None,
        "three timestamps cannot time five frames"
    );

    let mut both = Reader::default();
    both.absorb(&all);
    both.absorb(&[frame(3, 4000.0), frame(4, 5500.0)]);
    let joined = both.finish();
    assert_eq!(
        joined.duration_s(5),
        Some(5.5),
        "the second part's frames were not counted"
    );
    // And the record from the first part is not lost by absorbing a second.
    assert_eq!(joined.events.len(), 2);
    assert_eq!(joined.width, Some(512));
    assert!(joined
        .to_text("f.oir", 5)
        .expect("a record")
        .contains("\"T Dimension\"\t\"5, 0.000 - 5.500 [s], Interval FreeRun\""));
}

/// A few frames whose documents never got written must not cost the recording
/// its timing — but a recording whose later parts were never read must.
///
/// Both look the same from a count: fewer timestamps than frames. They are not
/// the same thing, and the difference is the whole duration. The acquisition
/// this was built against is a four-part, 7,213-frame recording missing three
/// documents from the middle; reading only its first file leaves 2,030 of them,
/// which is a quarter of the recording and looks exactly as complete.
///
/// What separates them is where the last timestamp lands. Frames arrive on a
/// fixed clock, so the last one belongs at `(frames - 1) * period` — and a
/// recording that stops a quarter of the way through does not put it there.
#[test]
fn documents_missing_from_the_middle_do_not_cost_the_duration() {
    const PERIOD: f64 = 133.333333;
    const FRAMES: usize = 7213;

    // Every frame but three, which are dropped from the middle exactly as the
    // real acquisition drops them — leaving three gaps of double the period.
    let mut reader = Reader::default();
    reader.absorb(&[properties()]);
    let mut docs = Vec::new();
    for i in 0..FRAMES {
        if [1000, 4000, 6000].contains(&i) {
            continue;
        }
        docs.push(frame(i, i as f64 * PERIOD));
    }
    reader.absorb(&docs);
    let r = reader.finish();

    assert_eq!(
        r.times.len(),
        FRAMES - 3,
        "the fixture is not what it claims"
    );
    let want = (FRAMES - 1) as f64 * PERIOD / 1000.0;
    let got = r
        .duration_s(FRAMES)
        .expect("three absent documents lost the timing");
    assert!(
        (got - want).abs() < 1e-6,
        "the recording is {want} s long, not {got}"
    );

    // The line that reaches tag 270, and with it the analysis downstream.
    let text = r.to_text("f.oir", FRAMES).expect("a record");
    assert!(
        text.contains("\"T Dimension\"\t\"7213, 0.000 - 961.600 [s]"),
        "{}",
        text.lines()
            .find(|l| l.contains("T Dimension"))
            .unwrap_or("(none)")
    );

    // Now the case that must still be refused: the same recording, timed only
    // as far as its first part reaches.
    let mut first_part = Reader::default();
    first_part.absorb(&[properties()]);
    first_part.absorb(&docs[..2030]);
    let part = first_part.finish();
    assert_eq!(
        part.duration_s(FRAMES),
        None,
        "a quarter of the timestamps were taken for the whole recording"
    );
    assert!(part
        .to_text("f.oir", FRAMES)
        .expect("a record")
        .contains("\"T Dimension\"\t\"7213 [T]\""));
}

/// Timestamps can be short of the frames and past the end of them at the same
/// time, and then neither counting nor taking the last one works.
///
/// This is the shape of a real 4,913-frame recording: two documents absent from
/// the middle, and one *extra* for the partial final frame the reader drops. It
/// carries 4,912 timestamps — one fewer than it has frames — while its last one
/// sits a whole frame *beyond* the last frame kept. Indexing runs out; taking
/// the last overshoots by 133 ms. Only the clock gets it right, and the file it
/// came from states the answer in its own sidecar: 654.933 s.
#[test]
fn timestamps_that_are_short_and_overshoot_at_once_still_give_the_duration() {
    const PERIOD: f64 = 133.333333;
    const FRAMES: usize = 4913;

    let mut reader = Reader::default();
    reader.absorb(&[properties()]);
    // Frame slots 0..=4913 — one more than the stack keeps — less two from the
    // middle.
    let docs: Vec<String> = (0..=FRAMES)
        .filter(|i| ![1500, 3000].contains(i))
        .map(|i| frame(i, i as f64 * PERIOD))
        .collect();
    reader.absorb(&docs);
    let r = reader.finish();

    assert_eq!(
        r.times.len(),
        FRAMES - 1,
        "the fixture is not what it claims"
    );
    let got = r.duration_s(FRAMES).expect("the duration was refused");
    assert!(
        (got - 654.933).abs() < 0.001,
        "the recording is 654.933 s, as its sidecar says; got {got}"
    );
    // Not the last timestamp, which belongs to the frame that was dropped.
    assert!(
        (got - r.times.last().unwrap() / 1000.0).abs() > 0.1,
        "the dropped frame's timestamp was taken for the recording's length"
    );
}

/// The stack decides the frame count, not the acquisition record.
///
/// An interrupted acquisition leaves a partial final frame that this reader
/// drops and the record still counts. Writing the record's number would state
/// one more frame than the file holds, and the frame interval read back out of
/// it — duration over gaps — would be wrong for every frame.
#[test]
fn the_frame_count_written_is_the_one_the_stack_ends_up_with() {
    let r = Record::parse(&documents());
    let text = r.to_text("f.oir", 2).expect("a record");
    assert!(
        text.contains("\"T Dimension\"\t\"2, 0.000 - 1.000 [s], Interval FreeRun\""),
        "{text}"
    );
    // Two frames, one gap of a second.
    assert_eq!(r.frame_interval_s(2), Some(1.0));
    // Three frames at 0, 1 and 2.5 s: 2.5 s over two gaps.
    assert_eq!(r.frame_interval_s(3), Some(1.25));
    // A single frame has no interval to state.
    assert_eq!(r.frame_interval_s(1), None);
}

/// Display lookup tables and drawing overlays are most of an OIR's XML by a
/// wide margin, and none of it describes the acquisition. Rejecting them on
/// the root element's name is what keeps them out of memory and out of tag 270.
#[test]
fn only_documents_that_describe_the_acquisition_are_kept() {
    for kept in [
        r#"<?xml version="1.0"?><lsmimage:imageProperties/>"#,
        r#"<?xml version="1.0"?><event:eventList/>"#,
        r#"<?xml version="1.0"?><base:imageDefinition/>"#,
        r#"<?xml version="1.0"?><lsmframe:frameProperties/>"#,
    ] {
        assert!(is_known_document(kept), "{kept}");
    }
    for dropped in [
        r#"<?xml version="1.0"?><lut:LUT/>"#,
        r#"<?xml version="1.0"?><overlay:contents/>"#,
        r#"<?xml version="1.0"?><annotation:annotationStore/>"#,
        "not xml at all",
    ] {
        assert!(!is_known_document(dropped), "{dropped}");
    }
}

/// `<a/>` is a complete element. Treating it as the start of one nests every
/// following sibling inside it, and the OIR schema is full of them —
/// `<commonphase:opticalResolution/>` sits directly above the pixel size.
#[test]
fn an_empty_element_does_not_swallow_the_siblings_after_it() {
    let el =
        El::parse(r#"<?xml version="1.0"?><root><empty/><after>7</after></root>"#).expect("parse");
    assert_eq!(
        el.num_at("after"),
        Some(7.0),
        "the sibling was nested inside"
    );

    // And the real shape it appears in: the fixture's first channel has one.
    let r = Record::parse(&documents());
    assert_eq!(r.pixel_x, Some(0.621480569402239));
}

/// A document that does not parse contributes nothing, exactly as a missing
/// one would. The alternative — failing the import — would throw away pixels
/// that are perfectly readable because a metadata block was truncated.
#[test]
fn a_malformed_document_is_skipped_rather_than_fatal() {
    let mut docs = documents();
    docs.insert(
        0,
        "<?xml version=\"1.0\"?><lsmimage:imageProperties>\0\u{1}".into(),
    );
    let r = Record::parse(&docs);
    assert_eq!(r.width, Some(512), "the good document should still be read");
    assert_eq!(r.events.len(), 2);
}

#[test]
fn the_acquisition_date_is_written_the_way_the_sidecar_writes_it() {
    for (iso, want) in [
        (
            "2025-07-28T15:36:14.925-04:00",
            "07/28/2025 03:36:14.925 PM",
        ),
        (
            "2025-02-10T18:49:39.124-05:00",
            "02/10/2025 06:49:39.124 PM",
        ),
        // Midnight and noon are where a 12-hour clock goes wrong.
        ("2025-01-01T00:00:00.000Z", "01/01/2025 12:00:00.000 AM"),
        ("2025-01-01T12:00:00.000Z", "01/01/2025 12:00:00.000 PM"),
        // No fractional part: the field keeps its width.
        ("2025-01-01T09:05:07", "01/01/2025 09:05:07.000 AM"),
    ] {
        assert_eq!(fluoview_date(iso).as_deref(), Some(want), "{iso}");
    }
    for junk in [
        "",
        "yesterday",
        "2025-07-28",
        "not-a-date-at-all!!",
        "9999-99-99T99:99:99",
    ] {
        assert_eq!(fluoview_date(junk), None, "{junk}");
    }
}

/// A file with nothing recognizable in it writes no description at all, rather
/// than a scaffold of empty section headers that looks like metadata.
#[test]
fn a_file_with_no_acquisition_record_produces_no_text() {
    assert!(Record::parse(&[]).to_text("f.oir", 3).is_none());
    let junk = vec![
        r#"<?xml version="1.0"?><lut:LUT><lut:intensity>3</lut:intensity></lut:LUT>"#.to_string(),
    ];
    assert!(Record::parse(&junk).to_text("f.oir", 3).is_none());
}

/// The bounds exist so a hostile file fails on a limit rather than on memory.
#[test]
fn an_absurdly_deep_document_is_refused_rather_than_followed() {
    let deep = format!(
        "<?xml version=\"1.0\"?><a>{}{}</a>",
        "<b>".repeat(MAX_DEPTH + 10),
        "</b>".repeat(MAX_DEPTH + 10)
    );
    assert!(El::parse(&deep).is_none());

    let wide = format!(
        "<?xml version=\"1.0\"?><a>{}</a>",
        "<b/>".repeat(MAX_ELEMENTS + 1)
    );
    assert!(El::parse(&wide).is_none());
}
