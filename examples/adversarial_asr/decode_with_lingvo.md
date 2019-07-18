# Decode

## Given the trained state of art lingvo asr model make a script to decode with it
Known that the given model in this project is trained by librispeech dataset. 

Decoding with this model for benchmarking my own ASR mdl

### Test on public dataset

* WER

| test id | DataSet | speaker type | sentence amount | WER |  
| --- | --- | --- | --- | --- |
| 001 | L2-Arctic-CN | nonnative | 600 | WER 30.26 [ 1710 / 5651, 257 ins, 146 del, 1307 sub ] |
| 002-1 | cmu-Arctic-US-dbl | native | 1130 | WER 5.19 [ 517 / 9971, 125 ins, 26 del, 366 sub ] |
| 002-2 | cmu-Arctic-US-slt | native | 1130 | WER 4.87 [ 486 / 9971, 119 ins, 15 del, 352 sub ] |
| 002-3 | cmu-Arctic-US-clb | native | 1130 | WER 4.81 [ 480 / 9971, 119 ins, 19 del, 342 sub ] |
| 002-4 | cmu-Arctic-US-rms | native | 1130 | WER 5.23 [ 521 / 9971, 155 ins, 15 del, 351 sub ] |
| 003 | cmu-Arctic-CANADIAN-jmk | native-accent | 1130 | WER 8.23 [ 821 / 9971, 155 ins, 109 del, 557 sub ] |
| 004 | cmu-Arctic-SCOTTISH-awb | native-accent | 1130 | WER 67.63 [ 6743 / 9971, 1050 ins, 932 del, 4761 sub ] |
| 005 | cmu-Arctic-INDIAN-ksp | nonnative | 1130 | WER 27.61 [ 2753 / 9971, 436 ins, 222 del, 2095 sub ] |
| 006 | Librispeech-test-clean | native | 2620 |  |
| 007 | Librispeech-test-other | native | 2940 |  |
