# Аудит данных (ТЗ 1.2)
Стадия: `full`; smoke: `False`. Только счётчики: тексты документов в аудит не попадают.

## Документы по источникам
| источник | документов | инъекции | чистые | из них E6-варианты | train | val | test | выпало по дедупу |
|---|---|---|---|---|---|---|---|---|
| bipia | 7438 | 7260 | 178 | 7082 | 0 | 1598 | 5840 | 318 |
| deep | 662 | 263 | 399 | 0 | 436 | 110 | 116 | 1 |
| dojo | 3840 | 1270 | 2570 | 0 | 0 | 90 | 2971 | 1021 |
| dyn | 2533 | 380 | 2153 | 0 | 0 | 0 | 2533 | 45 |
| notinject | 339 | 0 | 339 | 0 | 0 | 0 | 339 | 0 |
| para | 1073 | 840 | 233 | 0 | 0 | 0 | 1073 | 0 |

## Языки (langdetect, сид из конфига)
| источник | en | non-en | unk | топ кодов |
|---|---|---|---|---|
| bipia | 7300 | 138 | 0 | en:7300, id:46, de:46, sk:46 |
| deep | 355 | 307 | 1 | en:355, de:265, es:7, af:6, nl:5, id:4 |
| dojo | 3141 | 699 | 153 | en:3141, no:210, fr:177, unk:153, sv:44, ca:40 |
| dyn | 2337 | 196 | 138 | en:2337, unk:138, fr:34, no:20, vi:2, it:1 |
| notinject | 254 | 85 | 0 | en:254, zh-cn:73, ko:9, ru:1, es:1, bg:1 |
| para | 1073 | 0 | 0 | en:1073 |

Доля немецкого в deepset:

| часть | документов | de | доля de | en |
|---|---|---|---|---|
| test | 116 | 46 | 39.7 % | 65 |
| train | 546 | 219 | 40.1 % | 290 |
| all | 662 | 265 | 40.0 % | 355 |

## Длины нормализованного текста (символы)
| источник | min | медиана | среднее | max |
|---|---|---|---|---|
| bipia | 69 | 792 | 906.1 | 3524 |
| deep | 7 | 65 | 118.8 | 4545 |
| dojo | 2 | 327 | 1798.0 | 28543 |
| dyn | 2 | 202 | 579.8 | 6905 |
| notinject | 11 | 90 | 87.3 | 233 |
| para | 19 | 240 | 310.0 | 1248 |

## Окна (ТЗ 1.3)
| источник | окон | окон на документ | окна-инъекции | окна-чистые | исключено дедупом |
|---|---|---|---|---|---|
| bipia | 36322 | 4.88 | 8742 | 27580 | 1236 |
| deep | 777 | 1.17 | 372 | 405 | 1 |
| dojo | 37373 | 9.73 | 5303 | 32070 | 16519 |
| dyn | 8663 | 3.42 | 1199 | 7464 | 45 |
| notinject | 339 | 1.0 | 0 | 339 | 0 |
| para | 2031 | 1.89 | 1796 | 235 | 0 |

## Состав NotInject
| подмножество | Common Queries | Multilingual | Technique Queries | Virtual Creation | non-en |
|---|---|---|---|---|---|
| one | 58 | 25 | 16 | 14 | 25 |
| three | 19 | 29 | 41 | 24 | 29 |
| two | 49 | 30 | 30 | 4 | 31 |

## BIPIA (ТЗ 1.4)
| задача | строк в test.jsonl | уникальных контекстов | кластеров (почти-дубли слиты) | имён атак | строк атак | документов (основной тест) | документов E6 (доп.) |
|---|---|---|---|---|---|---|---|
| code | 50 | 50 | 50 | 10 | 50 | 100 | 1450 |
| email | 50 | 44 | 32 | 15 | 75 | 88 | 1936 |
| table | 100 | 84 | 84 | 15 | 75 | 168 | 3696 |

Контексты: val 33, test 133; позиции ['start', 'middle', 'end']; правило `middle`: начало предложения по regex `[.!?]+[кавычки]*\s+` (плюс смещение 0), выбранное RNG контекста.
- задача `abstract` выпала: context corpus not in the BIPIA repository (BLOCKERS B4)
- задача `qa` выпала: context corpus not in the BIPIA repository (BLOCKERS B4)

## Дедупликация (ТЗ 1.7)
- правило: Жаккар >= 0.8 по символьным 5-граммам, кандидаты MinHash LSH (128 перестановок, b=25, r=5, вероятность кандидата на пороге 0.999951), проверка точным Жаккаром
- окон в тесте: 73056, эталонных (train+val): 7685, исключено тестовых окон: 17801 (по источникам: {'bipia': 1236, 'deep': 1, 'dojo': 16519, 'dyn': 45}; пары источников: {'bipia->bipia': 1236, 'deep->deep': 1, 'dojo->dojo': 16519, 'dyn->dojo': 45})
- документов выпало: {'positives_all_excluded': 186, 'no_windows_left': 1067, 'bipia_pairs': 132, 'bipia_contexts': 4}; по источнику/варианту/метке: {'bipia/e6/1': 310, 'bipia/main/0': 4, 'bipia/main/1': 4, 'deep/main/0': 1, 'dojo/main/0': 1021, 'dyn/main/0': 45}

## Пулы негативов (ТЗ 1.8)
- p_val: 197 документов {'bipia': 38, 'deep': 69, 'dojo': 90}; цель 2000, не достигнута; deepset-часть: val
- p_test: 1085 документов {'bipia': 136, 'deep': 55, 'dojo': 106, 'dyn': 555, 'para': 233}; цель 2000, не достигнута

## Пересечение с обучающими данными промышленных детекторов (ТЗ 3.2)
Открытый обучающий набор PIGuard (`data/raw/piguard_train/train.json`, чтение журналировано как `external`): записей 76735, пустых 3, уникальных текстов 73520, окон при сканировании 197578. Правило: exact Jaccard >= 0.8 over character 5-gram shingles (dedup.jaccard, ТЗ 1.7); кандидаты — MinHash LSH (b=25, r=5).

Состав train.json по полю `source` (метка 0 / 1):

| тег PIGuard | 0 | 1 |
|---|---|---|
| Alpaca | 4000 | 0 |
| BIPIA | 558 | 558 |
| ChatGPT-Jailbreak-Prompts | 0 | 79 |
| InjecAgent | 0 | 111 |
| LLM Augmented set | 0 | 435 |
| Prompt-Injection-Mixed-Techniques | 0 | 1174 |
| Question Set | 643 | 1643 |
| StruQ | 0 | 20 |
| TaskTracker | 11386 | 3316 |
| awesome-chatgpt-prompts | 170 | 0 |
| chatbot_instruction_prompts | 16000 | 0 |
| grok-conversation-harmless | 4000 | 0 |
| hackaprompt-dataset | 0 | 5000 |
| jailbreak-classification | 517 | 527 |
| no_robots | 1500 | 0 |
| open-instruct | 12000 | 0 |
| over-defense | 762 | 0 |
| prompt-injections | 343 | 203 |
| safe-guard-prompt-injection | 5740 | 2496 |
| ultrachat_200k | 3000 | 0 |
| vigil-jailbreak-ada-002 | 0 | 104 |
| xtest-v2-copy | 450 | 0 |

Наши документы и окна с почти-дубликатом в train.json (по источнику, разбиению E1 и метке; метка документа для столбцов документов, метка окна (ТЗ 1.3) для столбцов окон — у документов E6 есть чистые окна контекста; `bipia_e6` — варианты E6):

| источник | split | метка | документов | совпало документов | доля | окон | совпало окон | доля окон | документов с совпавшим окном |
|---|---|---|---|---|---|---|---|---|---|
| bipia | test | 0 | 140 | 14 | 0.1 | 1170 | 55 | 0.047 | 22 |
| bipia | test | 1 | 140 | 5 | 0.0357 | 174 | 2 | 0.0115 | 17 |
| bipia | val | 0 | 38 | 7 | 0.1842 | 251 | 23 | 0.0916 | 10 |
| bipia | val | 1 | 38 | 4 | 0.1053 | 45 | 2 | 0.0444 | 8 |
| bipia_e6 | test | 0 | - | - | - | 21564 | 975 | 0.0452 | - |
| bipia_e6 | test | 1 | 5560 | 253 | 0.0455 | 6712 | 64 | 0.0095 | 826 |
| bipia_e6 | val | 0 | - | - | - | 4595 | 473 | 0.1029 | - |
| bipia_e6 | val | 1 | 1522 | 149 | 0.0979 | 1811 | 17 | 0.0094 | 383 |
| deep | test | 0 | 56 | 1 | 0.0179 | 58 | 1 | 0.0172 | 1 |
| deep | test | 1 | 60 | 3 | 0.05 | 75 | 6 | 0.08 | 3 |
| deep | train | 0 | 274 | 274 | 1.0 | 277 | 277 | 1.0 | 274 |
| deep | train | 1 | 162 | 162 | 1.0 | 249 | 249 | 1.0 | 162 |
| deep | val | 0 | 69 | 69 | 1.0 | 70 | 70 | 1.0 | 69 |
| deep | val | 1 | 41 | 41 | 1.0 | 48 | 48 | 1.0 | 41 |
| dojo | test | 0 | 1970 | 0 | 0.0 | 28044 | 0 | 0.0 | 0 |
| dojo | test | 1 | 1001 | 0 | 0.0 | 4226 | 0 | 0.0 | 0 |
| dojo | unused | 0 | 510 | 0 | 0.0 | 3687 | 0 | 0.0 | 0 |
| dojo | unused | 1 | 269 | 0 | 0.0 | 1077 | 0 | 0.0 | 0 |
| dojo | val | 0 | 90 | 0 | 0.0 | 339 | 0 | 0.0 | 0 |
| dyn | test | 0 | 2153 | 0 | 0.0 | 7464 | 0 | 0.0 | 0 |
| dyn | test | 1 | 380 | 0 | 0.0 | 1199 | 0 | 0.0 | 0 |
| notinject | test | 0 | 339 | 0 | 0.0 | 339 | 0 | 0.0 | 0 |
| para | test | 0 | 233 | 0 | 0.0 | 235 | 0 | 0.0 | 0 |
| para | test | 1 | 840 | 0 | 0.0 | 1796 | 0 | 0.0 | 0 |

Совпавшие документы по тегу PIGuard: bipia: {'BIPIA': 30}; bipia_e6: {'BIPIA': 402}; deep: {'ChatGPT-Jailbreak-Prompts': 4, 'Question Set': 3, 'awesome-chatgpt-prompts': 7, 'chatbot_instruction_prompts': 1, 'jailbreak-classification': 4, 'open-instruct': 1, 'prompt-injections': 547, 'safe-guard-prompt-injection': 10, 'vigil-jailbreak-ada-002': 4}

Точное вхождение (containment) против записей с тегом источника-двойника:
- deep vs ['prompt-injections'] (546 записей, 662 наших документов): наш документ внутри записи PIGuard — 554 (0.8369), запись PIGuard внутри нашего документа — 553 (0.8353); по split: {'test': {'documents': 116, 'ours_in_piguard': 8, 'piguard_in_ours': 7}, 'train': {'documents': 436, 'ours_in_piguard': 436, 'piguard_in_ours': 436}, 'val': {'documents': 110, 'ours_in_piguard': 110, 'piguard_in_ours': 110}}
- bipia vs ['BIPIA'] (794 записей, 356 наших документов): наш документ внутри записи PIGuard — 43 (0.1208), запись PIGuard внутри нашего документа — 70 (0.1966); по split: {'val': {'documents': 76, 'ours_in_piguard': 14, 'piguard_in_ours': 20}, 'test': {'documents': 280, 'ours_in_piguard': 29, 'piguard_in_ours': 50}}
- dojo vs ['InjecAgent'] (111 записей, 1270 наших документов): наш документ внутри записи PIGuard — 0 (0.0), запись PIGuard внутри нашего документа — 0 (0.0); по split: {'test': {'documents': 1001, 'ours_in_piguard': 0, 'piguard_in_ours': 0}, 'unused': {'documents': 269, 'ours_in_piguard': 0, 'piguard_in_ours': 0}}
- dyn vs ['InjecAgent'] (111 записей, 380 наших документов): наш документ внутри записи PIGuard — 0 (0.0), запись PIGuard внутри нашего документа — 0 (0.0); по split: {'test': {'documents': 380, 'ours_in_piguard': 0, 'piguard_in_ours': 0}}

Ограничение: ТЗ 1.3 windows (size 256, stride 192) of both sides; alignment-sensitive, a lower bound.

Карточки моделей (статично):
- `protectai_v2` (protectai/deberta-v3-base-prompt-injection-v2): обучающие наборы по карточке — natolambert/xstest-v2-copy, VMware/open-instruct, alespalla/chatbot_instruction_prompts, HuggingFaceH4/grok-conversation-harmless, Harelix/Prompt-Injection-Mixed-Techniques-2024, OpenSafetyLab/Salad-Data, jackhhao/jailbreak-classification; пересечение по именам: none of the listed datasets is deepset/prompt-injections, BIPIA, NotInject, AgentDojo or AgentDyn; the card's list is not exhaustive, the open training data is not published, so no measured overlap is possible
- `piguard` (leolee99/PIGuard): обучающие наборы по карточке — the authors' own train.json (data/raw/piguard_train/train.json; composition measured below by its `source` tag); пересечение по именам: train.json tags 546 records `prompt-injections` (deepset train: 343 benign / 203 injections) and 1116 records `BIPIA`; NotInject is the authors' own over-defense benchmark
- `prompt_guard_2` (meta-llama/Llama-Prompt-Guard-2-86M): обучающие наборы по карточке — not disclosed on the card; optional model, skipped without HF access (BLOCKERS B2); пересечение по именам: unknown

Угрозы валидности (ТЗ 3.2):
- BIPIA is part of the PIGuard authors' test set and 1116 BIPIA-tagged records are in the PIGuard open training set
- NotInject, PIGuard and AgentDyn share a first author: the over-defense set and the agent benchmark were built by the team that trained one of the comparators
- deepset/prompt-injections train (546 records, 343/203) is in the PIGuard open training set under the tag `prompt-injections`; E1 trains FlyGuard on the same 546 documents, so on deepset both sides saw the train split
- ProtectAI v2's card lists datasets by name only; an unmeasured overlap with deepset or BIPIA cannot be excluded
- No direction of the shift is assumed (ТЗ 3.2); the counts above are reported, not corrected for

## Пропущенные источники и замечания
- dojo: 3840 documents from 1046 episodes (data/traces/agentdojo)
- dyn: 2533 documents from 256 episodes (data/traces/agentdyn)
- para: 1073 documents
