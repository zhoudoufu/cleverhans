# Decode

## Given the trained state of art lingvo asr model make a script to decode with it
Known that the given model in this project is trained by librispeech dataset. 

Decoding with this model for benchmarking my own ASR mdl

### Test on public dataset

* WER

| test id | DataSet | speaker type | sentence amount | WER |  
| --- | --- | --- | --- | --- |
| 001 | L2-Arctic-CN | nonnative | 600 | WER 30.26 [ 1710 / 5651, 257 ins, 146 del, 1307 sub ] |
| 002 | cmu-Arctic-US | native |  |  |
| 003 | cmu-Arctic-CANADIAN | native |  |  |
| 004 | cmu-Arctic-SCOTTISH | native |  |  |
| 005 | cmu-Arctic-INDIAN | nonnative |  |  |
| 006 | Librispeech-test-clean | native |  |  |
| 007 | Librispeech-test-other | native |  |  |
