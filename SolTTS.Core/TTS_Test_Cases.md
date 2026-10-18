# SolTTS Multilingual Comprehensive Test Cases (ZH/JA/EN)

This document provides comprehensive test cases for validating SolTTS synthesis quality, language-switching stability, and G2P handling.

---

## 1. Basic Functionality

Verifies correct synthesis of plain text in each language.

| ID | Input | Language | Expected result |
|----|------|------|---------|
| B1 | `你好，欢迎使用SolTTS。` | ZH | Pure Chinese, natural pauses, clear audio |
| B2 | `こんにちは、SolTTSへようこそ。` | JA | Pure Japanese, correct accent, natural intonation |
| B3 | `Hello, welcome to SolTTS.` | EN | Pure English, native pronunciation |
| B4 | `我是AI助手。` | ZH | ZH-EN mixed (acronym) |
| B5 | `私はAIアシスタントです。` | JA | JA-EN mixed (acronym, katakana) |

---

## 2. Multilingual Mixing (Code-Switching)

Verifies coherence when switching languages within a sentence or paragraph.

### 2.1 Bilingual Mixing

| ID | Input | Combination | Challenge |
|----|------|------|------|
| M1 | `这个feature很不错，我们可以加进PR。` | ZH-EN | English term embedded in a sentence |
| M2 | `今日のランチはHamburgerを食べました。` | JA-EN | JA-EN mix, verb conjugation |
| M3 | `我刚才在听《さくら》，旋律非常优美。` | ZH-JA | ZH-JA mix, Japanese song title |
| M4 | `OpenAI发布了最新的model。` | ZH-EN | Letter word + Chinese |
| M5 | `駅の近くにStarbucksがあります。` | JA-EN | JA-EN mix, place/store names |

### 2.2 Trilingual Mixing (ZH-JA-EN)

| ID | Input | Challenge |
|----|------|------|
| T1 | `我在GitHub上看到一个日本のプロジェクト，非常Interesting。` | Rapid trilingual switching with proper nouns |
| T2 | `使用Python处理日本語のテキスト是一个Common task。` | Technical terms + Japanese particles + English nouns |
| T3 | `今天的感觉是Happy，明日も頑張りましょう，Keep going!` | Cross-language switching of emotional expression |

---

## 3. Edge Cases and Special Handling

### 3.1 Acronyms, Capitalization, and Proper Nouns

| ID | Input | Challenge |
|----|------|------|
| A1 | `TTS, AI, LLM, GPT-4, ONNX` | Reading common acronyms |
| A2 | `SolTTS, TensorFlow, PyTorch` | Proper-noun stress |
| A3 | `NHK, J-POP, AKB48` | Japanese-specific acronyms |
| A4 | `USA, UK, UN` | Country/organization acronyms |

### 3.2 Numeric & Symbols

| ID | Input | Domain | Challenge |
|----|------|------|------|
| N1 | `2024年1月15日 14:30` | Time and date | Difference between Chinese and Japanese readings |
| N2 | `123,456.78元 / ￥12,800` | Currency | Thousands separators and units |
| N3 | `成功率 95.5%，增长了 3倍。` | Percentages/multiples | Symbol handling |
| N4 | `1つ、2つ、3つ、1人、2人、3人` | Japanese counters | Kun'yomi / on'yomi switching (critical test) |
| N5 | `第1名 / 1位 / 1番目` | Ordinals | Context disambiguation |
| N6 | `3.1415926` | Long decimals | Continuity |

### 3.3 Punctuation and Special Symbols

| ID | Input | Challenge |
|----|------|------|
| S1 | `Hello, world!` | Standard English punctuation |
| S2 | `こんにちは、世界。` | Japanese punctuation (、。) |
| S3 | `你好……（沉默）` | Ellipsis and parentheses |
| S4 | `C++ / C# / .NET` | Reading programming symbols |
| S5 | `test@example.com` | Email symbol (@) |
| S6 | `https://github.com` | URL symbols (:/.) |
| S7 | `!!! ???` | Tone of repeated punctuation |

---

## 4. Japanese Specific Tests

Focuses on Japanese G2P (OpenJTalk) behavior.

### 4.1 Homographs

| ID | Input | Challenge |
|----|------|------|
| J1 | `明日、東京へ行きます。` | 「明日」(ashita) |
| J2 | `明日の予定を確認します。` | 「明日」(asu / ashita) |
| J3 | `銀行へ行く。` | 「行」(kou) |
| J4 | `三日間、旅行に行った。` | 「行」(it) |
| J5 | `名前は山田です。` | Place-name/person-name readings |
| J6 | `昨日、今日、明日` | Time-word combinations |

### 4.2 Phonetics: Sokuon, Hatsuon, and Long Vowels

| ID | Input | Challenge |
|----|------|------|
| P1 | `いってきます / いらっしゃい` | Sokuon (っ) |
| P2 | `しんぶん / ほん / おんな` | Hatsuon (ん) in different positions |
| P3 | `お母さん / おじいさん / クーラー` | Long-vowel markers |

### 4.3 Loanwords in Katakana

| ID | Input | Challenge |
|----|------|------|
| K1 | `コンピューター / インターネット` | Long katakana words |
| K2 | `スマートフォン / ログイン` | Compound loanwords |

---

## 5. Extreme and Stress Tests

| ID | Input | Challenge |
|----|------|------|
| E1 | `AAAAAAAAAAAAAAAAAAAA` | Very long repeated letters |
| E2 | `笑笑笑笑笑笑笑笑笑笑` | Repeated characters |
| E3 | `abc123def456ghi789` | Mixed characters without spaces |
| E4 | `这是一个非常非常非常非常非常非常长的句子，目的是为了测试在长文本输入时系统是否会崩溃或者出现语调断层。` | Long Chinese stability |
| E5 | `この文章はテスト用です。長い文章を入力しても、システムが正常に動作し続けるかどうかを確認するためのものです。` | Long Japanese stability |
| E6 | A 500+ character mixed trilingual long text (e.g., a technical article summary) | Memory and streaming output stress |

---

## 6. Semantic Context

| ID | Input | Expected / Challenge |
|----|------|---------|
| C1 | `那个人真的很“牛”。` | Stress/emphasis inside quotation marks |
| C2 | `Wait... what?` | Pauses and intonation |
| C3 | `彼は「行かない」と言った。` | Japanese quotation marks (「」) pauses |

---

## Known Issues and TODOs

- [ ] Pronunciation optimization for single English letters (e.g., `A`, `I`)
- [ ] Natural reading of programming symbols (e.g., `C#` should be "C-Sharp", not "C-Hash")
- [ ] Accuracy of Japanese counter words with complex numbers
- [ ] Intonation normalization for trilingual mixing
