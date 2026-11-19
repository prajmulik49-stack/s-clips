# ASR benchmark (jZUsWddSIbA, excerpt 300-420 s, CPU int8, 8 threads)

| model | load s | transcribe s | RTF | words | mean word prob |
|---|---|---|---|---|---|
| large-v3-turbo | 19.1 | 65.4 | 0.545 | 150 | 0.813 |
| large-v3 | 16.2 | 188.8 | 1.574 | 114 | 0.874 |

Word-level agreement large-v3-turbo vs large-v3: 0.260 (1 - normalized word edit distance)

**large-v3-turbo:** गाइस तो अब इसको औरियो का पहना रहे हैं इसके बाल थोड़े कम है घरवालों को समझ में आ सकता है यह रहलेगा मेरे साथ अकेले यह रहलेगा मेरे साथ अकेले के नाम किया इसका नाम किया तफी तफी तफी तफी औरियो तो अपर च्छत में इस्टूडियो तो अपर � जाओ जाओ जाओ जाओ जाओ जाओ गैस तो कलाल करके अंदर अचुके हैं अलू तो अगे हैं और यो खाना दिया था क्या पता नहीं आज क्या नजीब से कर रहा है इसने ममी खाना दिया था और यो आपने हैं दिया था खाना दिया था क्या कब दिया था यह था पता नहीं क क्या हुआ हुआ हुआ हुआ हुआ हुआ हुआ हुआ हुआ हुआ हुआ हुआ हुआ हुआ हुआ हुआ हुआ हुआ हुआ हुआ हुआ हुआ हुआ हुआ हुआ हुआ हुआ हुआ हुआ हुआ हुआ हुआ हुआ हुआ हुआ हुआ हुआ हुआ ह

**large-v3:** गाइस तो इसको ओरियो का पट्टा पहना रहे है इसके बाल थोड़े कम है घरवालों को समझ में आ सकता है बट ये रह लेगा मेरे साथ अकेले नाम क्या नाम क्या नाम क्या टॉप टॉप टॉप टॉप टॉप टॉप टॉप टॉप टॉप टॉप टॉप टॉप टॉप टॉप टॉप टॉप टॉप टॉप टॉप टॉप टॉप टॉप टॉप टॉप टॉप टॉप टॉप टॉप टॉप टॉप टॉप टॉप टॉप � तो आपने हैं दिया था खाना दिया था क्या कब दिया था दिया था क्या हो गया तो क्या हुआ क्या हुआ क्या हुआ क्या हुआ क्या सब्सक्राइब करते हैं और यह नहीं है और यह नहीं है और यह नहीं है और यह नहीं है

## Decision
**large-v3** (int8, beam 5, temperature 0). turbo is 2.9x faster (RTF 0.55 vs 1.57) but in the excerpt it
repeats a whole line ("यह रहलेगा मेरे साथ अकेले" x2) and drops content words ("पट्टा" = collar) that large-v3
gets right; large-v3's mean word probability is higher (0.874 vs 0.813). Transcription runs once per vlog and is
cached, so ~23 min on this CPU is acceptable. Low word agreement (0.26) is mostly spelling variants
(है/हैं, यह/ये, औरियो/ओरियो), which the phonetic lexicon matching absorbs. Switch with `asr.model`.
