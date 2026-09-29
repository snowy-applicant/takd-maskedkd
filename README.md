# TAKD × MaskedKD: 연산량 대비 학습 효율

**질문.** Teacher assistant(TA)를 쓰는 KD(TAKD)에 MaskedKD를 적용하면, 기존 HKD(직접 KD)보다 늘어나는 연산량 대비 학습 효율을 높일 수 있을까?

| 군 | T → A (1단계) | A → S / T → S (2단계) | 설명 |
|---|---|---|---|
| A | masking | full | |
| B | full | masking | |
| C | masking | masking | |
| D | full | full | 일반 TAKD |
| E | – | masking (T → S) | TA 없는 MaskedKD |
| F | – | full (T → S) | TA 없는 KD, **기준** |

- Teacher: **DeiT-B** (ImageNet 가중치에서 시작해 COCO single에 fine-tune)
- Teacher assistant: **DeiT-S (width ×2)** — Passalis et al. 논문의 보조 교사 규칙("학생과 같은 구조, 층마다 뉴런 2배")
- Student: **DeiT-S** (ImageNet 가중치에서 시작)
- 데이터: [kshs-aimlab-benchmarks](https://github.com/jeehoo0507/kshs-aimlab-benchmarks)의 **COCO single** (10 클래스, train 3210 / val 530 / test 297)
- masking ratio 0.5 (teacher는 196 패치 중 98개만 보고, 가려진 토큰은 **제거**)
- 지표: 학습 FLOPs, 달성 FLOP/s(초당 연산 속도), 학습 시간, test macro accuracy → **A–E를 F로 나눈 비율**을 seed 3개(0, 1, 2)에 대해 평균 ± 표준편차로 정리

같은 실험을 손실 함수만 바꿔 두 번 합니다(`suite`).

| suite | 증류 손실 | 목적 |
|---|---|---|
| `hkd` (주 실험) | Passalis et al. (CVPR 2020)의 information-flow KD + CE | 계획서의 HKD/TA 정의를 그대로 따름 |
| `logit` (강건성 확인) | 0.5·CE + 0.5·KL(T=1) — MaskedKD·기준 레포와 같은 logit KD | 결론이 손실 선택에 달려 있는지 확인 (Mirzadeh et al.의 원래 TAKD 형태) |

---

## 서버에서 실행하기

모든 명령은 레포 폴더 안에서 실행합니다. **sudo가 필요 없고, 시스템 Python(3.14.7)·`$HOME`·셸 설정 파일을 건드리지 않습니다.** uv, CPython 3.12, 가상환경, 모든 캐시가 레포 안(`.tools/`, `.cache/`, `.venv/`, `.tmp/`)에만 생기며, 지우려면 폴더를 삭제하면 끝입니다.

```bash
git clone https://github.com/snowy-applicant/takd-maskedkd.git
cd takd-maskedkd
```

```bash
bash scripts/setup.sh
```
uv 0.12.20을 `.tools/uv/bin`에(`UV_UNMANAGED_INSTALL`: 셸 설정·receipt 기록 없음), uv가 관리하는 CPython 3.12를 `.tools/python`에 설치하고, `uv.lock` 그대로 `.venv`를 만듭니다. PyTorch 2.14.0은 NVIDIA 드라이버를 보고 고릅니다(드라이버 ≥ 580 → `cu130`, 525–579 → `cu126`). 강제로 고르려면 `TAKD_TORCH_EXTRA=cu126 bash scripts/setup.sh`.

```bash
bash scripts/smoke_test.sh
```
다운로드 없이 수 분 안에 끝납니다. 단위 테스트와, 작은 모델·합성 데이터로 전체 파이프라인(teacher → 두 TA → A–F × seed 2개 × 두 suite → 보고서)을 돌립니다.

```bash
bash scripts/prepare_data.sh
```
COCO 2017 annotation과 조건에 맞는 이미지만 받아 기준 레포와 **같은 규칙·같은 split**으로 COCO single을 만들고 검증합니다. 이어서 224×224 캐시(`data/cache/coco_single_224.pt`, 약 0.6 GB)를 만들고 공식 DeiT-S/DeiT-B ImageNet 가중치를 받습니다. manifest SHA-256이 기준 레포 결과(`072f811c…`)와 같은지 출력합니다. 중단되면 다시 실행하면 이어서 합니다. 이미 데이터가 있으면 `--data-root /절대경로/coco_single`.

```bash
bash scripts/run_all.sh
```
두 suite 전체(teacher 1회 + suite마다 24 run)를 **백그라운드로 순차 실행**합니다. SSH를 끊어도 계속됩니다.

| 명령 | 용도 |
|---|---|
| `tail -f outputs/logs/latest.log` | 진행 로그 |
| `bash scripts/status.sh` | run별 epoch 진행, runner 상태, GPU 사용률 |
| `bash scripts/report.sh` | 지금까지 끝난 run으로 `results/` 표·그림 갱신 (학습 중에도 가능) |
| `bash scripts/plan.sh` | 실행 순서와 군별 해석적 FLOPs (GPU 불필요) |
| `bash scripts/stop.sh` | 중지. 같은 `run_all.sh`를 다시 실행하면 이어서 학습 (5 epoch마다 체크포인트) |
| `bash scripts/run_all.sh --suite hkd` | 한 suite만 |
| `bash scripts/run_all.sh --foreground` | tmux 안 등에서 붙어서 실행 |

`takd.cli run`의 옵션은 그대로 넘어갑니다: `--seeds 0`, `--groups F E D`, `--suite logit` 등.

**예상 시간.** RTX A5000에서 두 suite 합계 약 10–20시간입니다(달성 TFLOP/s에 따라 다름). 순서가 teacher → seed 0의 F, E → TA → … 이라 처음 한 시간 안에 첫 비교(F vs E)가 나옵니다. `bash scripts/plan.sh`가 실행 순서와 군별 FLOPs를 미리 보여 줍니다(`prepare_data.sh` 끝에도 출력됨).

**GPU 공유 주의.** 학습 시간이 핵심 지표라서 run을 동시에 돌리지 않고 순차 실행합니다. 다른 사람이 GPU를 쓰고 있으면 run 시작 시 경고를 남기며, 그 run의 시간은 부풀려질 수 있습니다(FLOPs와 정확도는 영향 없음). 다른 사용자의 작업은 절대 중단하지 않습니다.

---

## 결과 파일 (`results/<suite>/`)

| 파일 | 내용 |
|---|---|
| `SUMMARY.md` | **F 대비 비율 표**(FLOPs, 학습 시간, FLOP/s, 정확도, Δ정확도, 정확도/FLOPs, 목표 도달 FLOPs)와 절대값, teacher/TA 정확도 |
| `ratios.csv` | 위 비율들의 seed 평균·표준편차 |
| `groups.csv` | (군, seed)마다 1단계 + 2단계 비용과 정확도 |
| `summary.csv` | 군별 절대값의 평균·표준편차 |
| `runs.csv` | run 하나당 한 줄: 정확도, FLOPs, 시간, 최대 VRAM, 해시(데이터·teacher·체크포인트·코드), 환경 |
| `epochs.csv` | 모든 run의 epoch별 손실·val 정확도·시간·누적 FLOPs |
| `per_class.csv` | 클래스별 test 정확도 |
| `mask_validation.csv`, `mask_test.csv` | masking 진단: 같은 teacher의 full/masked 정확도, 예측 불일치, KL, 전경 패치 recall/precision, 선택 패치 겹침 |
| `fig_accuracy_vs_flops.png` | 누적 학습 FLOPs(1단계 포함) 대비 val 정확도 곡선 |
| `fig_ratios.png`, `fig_time_split.png` | F 대비 비율, 학습 시간 중 teacher forward 비중 |

run별 원자료는 `outputs/<suite>/<run>/`(`result.json`, `history.json`, `train.log`, `best.pt`)에 있습니다. 공유 teacher는 `outputs/teacher/`입니다.

---

## 실험 설계와 근거

### 데이터 — COCO single
기준 레포의 `datasets/coco_single/prepare.py`와 `scripts/check_assets.py`를 그대로 가져왔습니다(선택 규칙·split seed `20260922`·manifest 형식 동일, curl이 없을 때 urllib로 받는 부분만 추가). 이미지당 주석 객체가 정확히 하나인 10 클래스 분류이고 주 지표는 **macro accuracy**입니다. 변환도 기준과 같습니다: 이미지 전체를 224×224 bicubic으로 resize, 학습 때만 좌우 반전.

기준 runner는 JPEG를 `num_workers=0`으로 디코딩해 입력 파이프라인에 묶여 있었습니다(A5000에서 약 76 img/s). 그러면 masking으로 줄어든 teacher 연산이 학습 시간에 드러나지 않습니다. 그래서 resize 결과를 uint8로 한 번 캐시해 **데이터 전체를 GPU에 올려 두고**, 매 step은 GPU에서 반전·정규화만 합니다(결과 이미지는 기준과 동일).

### 모델
| 역할 | 구조 | 파라미터 | forward (224², 197 토큰) | 초기화 |
|---|---|---|---|---|
| Teacher | DeiT-B: dim 768, depth 12, heads 12 | 86.6 M | 35.2 GFLOPs (17.6 GMACs) | 공식 ImageNet 가중치 → COCO single에서 30 epoch fine-tune (CE) |
| TA | DeiT-S ×2: dim 768, depth 12, heads 12, MLP 3072 | 86.6 M | 35.2 GFLOPs | ImageNet DeiT-S를 폭 2배로 **함수 보존 확장**(아래) |
| Student | DeiT-S: dim 384, depth 12, heads 6 | 22.1 M | 9.2 GFLOPs | 공식 ImageNet 가중치 |

**중요: DeiT-S의 폭을 2배로 하면 DeiT-B와 크기가 같습니다.** 논문 규칙("모든 층의 뉴런 수 2배")을 ViT에 적용하면 embedding 384→768, MLP 1536→3072이 되고, head 폭을 64로 유지하면 head 수가 6→12가 되어 DeiT-B와 구조가 같아집니다(head 수 6·head 폭 128로 해도 파라미터·FLOPs는 같음). 따라서 계획서의 "teacher가 TA보다 훨씬 크다"는 전제가 성립하지 않고, 예상 순서 C<A<B<D는 해석적으로 **C < A = B < D**가 됩니다(아래 예측 표). 이 점이 이번 실험에서 가장 먼저 짚어야 할 결과입니다. 다른 폭을 시험하려면 `takd/models.py`의 `ARCHS`에 구조를 추가하고 config의 `archs.assistant`를 바꾸면 됩니다.

**TA 초기화.** 논문은 보조 교사의 초기화를 명시하지 않습니다(공개 코드는 CE로 먼저 학습한 보조 교사에서 시작). 이 실험의 student는 기준 레포처럼 ImageNet 가중치에서 시작하므로, TA도 같은 출발 지식을 갖게 하려고 ImageNet DeiT-S를 Net2WiderNet 방식으로 넓혔습니다. residual stream·q/k/v(=head)·MLP 은닉 유닛을 모두 두 벌 복사하고, 복사된 입력을 읽는 가중치는 반으로 나눕니다. LayerNorm 입력을 통째로 복제하면 평균·분산이 그대로라서 **확장 직후 TA는 DeiT-S와 정확히 같은 함수**입니다(테스트로 확인). 복사본끼리의 대칭은 경사하강(특히 Adam)으로는 깨지지 않으므로 가중치 행렬에 표준편차의 1%인 잡음을 더합니다(`widen_noise`). 대안: `"init": {"assistant": "imagenet"}`(DeiT-B 가중치, 구조가 같아서 가능) 또는 `"scratch"`.

### 증류 손실
모든 단계는 `L = w_ce·CE(label smoothing 0.1) + w_logit·T²·KL + w_flow·Σ_l α_l(epoch)·PKT_l` 하나로 표현됩니다.

**`hkd` suite — Passalis, Tzelepi, Tefas, *Heterogeneous Knowledge Distillation using Information Flow Modeling*, CVPR 2020**
- 층 쌍마다 Eq. 7: 배치 안의 조건부 확률 `p_{j|i} = K(x_i,x_j)/Σ_{k≠i} K(x_i,x_k)`를 cosine 커널과 Student-t 커널(d=1)로 만들고, teacher와 student 분포의 **Jeffreys divergence**(대칭 KL)를 더합니다. B×B 행렬만 비교하므로 폭이 달라도(768 vs 384) projector가 필요 없습니다.
- Eq. 10 critical period: 마지막 표현(최종 정규화 CLS, head 입력)의 가중치는 항상 1, 중간 층은 `100·0.7^k` (k = 0부터 세는 epoch). 13 epoch 이후 1보다 작아집니다.
- 층 대응: DeiT-B·TA·DeiT-S 모두 12 블록이므로 1:1 대응이 자연스럽습니다. 논문 CNN의 "블록 3개 + 마지막 FC" 구조에 맞춰 블록 **3, 6, 9의 CLS 토큰 + 최종 CLS**를 씁니다. CLS처럼 샘플 단위로 모은 표현은 teacher가 토큰 절반만 봐도 정의되므로 masking과 함께 쓸 수 있습니다.
- **T → A**: 논문 부록 A.2대로 teacher의 마지막 표현만 PKT로 전달합니다(중간 층·critical period 없음).
- **→ S (A → S와 직접 T → S)**: 모든 탭 층을 1:1로 전달하고 critical period를 적용합니다. F/E(직접 KD)와 A–D의 2단계가 **같은 손실**이라 TA 유무만 다릅니다.
- 분류 문제이므로 논문 Table 2처럼 CE를 더합니다(가중치 1; 논문에 명시 없음).
- 공개 코드와의 차이(논문 식을 따름): 대각선(i=j) 제외. 정규화는 j에 대해 합, i에 대해 평균("batchmean")이라 CE와 크기가 비교 가능합니다.

**`logit` suite** — MaskedKD 공식 코드·기준 runner의 손실 `0.5·CE + 0.5·KL(T=1)`을 두 단계 모두에 씁니다.

### MaskedKD (Son et al., ECCV 2024) — 공식 코드 `effl-lab/MaskedKD@96d052d`와 동일
- 각 단계의 **학생**(1단계에서는 TA, 2단계에서는 student)이 전체 이미지로 먼저 forward하고, 마지막 블록의 CLS 행 attention(head 평균)으로 상위 98개 패치를 고릅니다(같은 step, 현재 가중치, 기울기 없음).
- teacher는 모든 패치를 embedding하고 position embedding을 더한 **뒤** 고른 토큰만 모아(CLS는 항상 유지) forward합니다. 가려진 토큰은 mask token으로 바꾸지 않고 **제거**합니다.
- 학생은 항상 전체 이미지를 봅니다.

### 학습 설정 (기준 runner `configs/reference.json`을 따름)
AdamW(lr 5e-5, weight decay 0.05; bias·norm·pos_embed·cls_token 제외), 5 epoch linear warm-up 후 cosine으로 1e-6까지, batch 128, drop path 0.1, gradient clipping 1.0, label smoothing 0.1, 좌우 반전만 augmentation. Teacher 30 epoch, TA·student 100 epoch. checkpoint는 **validation macro accuracy로만** 고르고(동점이면 이른 epoch), test는 선택 후에 한 번 평가합니다(마지막 epoch test도 따로 기록).

기준 runner와 다른 점과 이유:
- batch 128을 누적 없이 한 번에(기준은 32×4). PKT는 배치 안의 유사도를 쓰므로 배치 전체가 필요합니다. 각 run은 시작할 때 실제 모델·optimizer 상태로 한 step을 돌려 VRAM을 확인합니다. 부족하면 CE·logit KD만 쓰는 run(teacher, `logit` suite)은 micro-batch를 절반으로 줄여 누적하고(목적함수가 정확히 같음) 그 값을 기록하며, PKT를 쓰는 `hkd` run은 목적함수가 바뀌므로 **멈추고** GPU를 비운 뒤 다시 실행하라고 알립니다(`"allow_pkt_micro_batch": true`로 허용 가능). 예상 최대 사용량은 TA 학습(86M 모델, batch 128, bf16)이 가장 커서 약 12–16 GiB입니다.
- 손실(PKT 커널, KL, CE)과 MaskedKD 패치 선택용 CLS attention은 bf16 autocast 밖에서 **fp32**로 계산합니다. autocast 안에서는 float32로 바꿔도 행렬곱이 bf16으로 다시 내려가기 때문입니다.
- 마지막 불완전 배치(3210 = 25×128 + 10)는 버립니다(같은 이유).
- bf16 autocast(A5000 지원, GradScaler 불필요)와 SDPA(flash) attention. `torch.compile`은 쓰지 않습니다 — Triton이 시스템 C 컴파일러를 요구하는데 초기화된 서버에 없을 수 있기 때문입니다.
- 공유 teacher 한 개를 두 suite가 함께 씁니다(CE만으로 학습하므로 suite와 무관).

### 지표 정의
- **FLOPs**: 해석적 *모델* FLOPs. 곱-덧셈 1회 = 2 FLOPs, 행렬곱(linear, patch-embedding conv, attention의 두 곱)만 셉니다. backward = forward×2(입력 기울기가 필요 없는 patch embedding은 ×1). 한 step = 학생 forward+backward + teacher forward(masked면 98+1 토큰, patch embedding은 196 패치 전부) + CLS attention 한 줄. 이 식은 torch `FlopCounterMode`와 **정확히 일치**함을 테스트로 확인했고, 각 run 시작 시 GPU에서 측정한 값도 `flops_per_sample_measured`로 남깁니다(flash attention의 재계산까지 세므로 약간 큼). 참고로 MaskedKD 논문 표는 MAC 단위(= FLOPs/2)입니다.
- **달성 FLOP/s (초당 연산 속도)**: 학습 FLOPs ÷ 학습 시간(TFLOP/s). 계획서의 "FLOPs(초당 연산 속도)"는 이 값으로, 총 연산량은 위의 FLOPs로 둘 다 보고합니다.
- **학습 시간**: 최적화 루프만(epoch 경계에서 CUDA 동기화; 평가·진단·체크포인트 제외). teacher forward 시간은 CUDA event로 따로 잽니다. `wall_seconds`는 평가·체크포인트 저장을 포함하되, masked run에서만 도는 masking 진단(`probe_seconds`)은 비교가 공정하도록 뺍니다. run 시작 시 다른 GPU 프로세스 수를 `gpu_other_processes`로 기록하고, 있었다면 SUMMARY.md에 경고를 붙입니다.
- **정확도**: validation으로 고른 checkpoint의 test macro accuracy(주 지표)와 overall accuracy.
- **비용 합산**: TA를 쓰는 군은 1단계(T→A)와 2단계를 더합니다. A와 C는 같은 masked TA를, B와 D는 같은 full TA를 공유해 실제로는 seed마다 TA를 두 개만 학습하지만, 각 군에는 TA 비용 전체를 매깁니다(그 방법을 단독으로 실행할 때의 비용). 모든 군이 공유하는 teacher fine-tuning은 제외하고 따로 보고합니다.
- **비율**: 같은 seed의 F로 나눈 뒤 seed 평균 ± 표본표준편차. 추가로 `정확도비/FLOPs비`와 **목표 도달 연산량**(val macro accuracy가 같은 seed F 최고값의 99%에 처음 닿을 때까지의 누적 FLOPs, 1단계 포함)을 계산합니다.

### 해석적 예측 (`bash scripts/plan.sh`, seed 하나, epoch당 3200 샘플)

| 군 | 학습 FLOPs | F 대비 |
|---|---|---|
| A | 59.25 PFLOP | 2.958 |
| B | 59.25 PFLOP | 2.958 |
| C | 53.58 PFLOP | 2.675 |
| D | 64.92 PFLOP | 3.241 |
| E | 14.36 PFLOP | 0.717 |
| F | 20.03 PFLOP | 1.000 |

1단계(T→A)의 비용은 TA 자신의 forward+backward(한 샘플 105 GFLOPs)가 대부분이고 masking이 줄이는 teacher forward(35→17 GFLOPs)는 일부라서, 1단계 masking의 절감률은 약 13%입니다. 2단계에서는 teacher forward 비중이 커서 masking이 약 28%를 줄입니다. TA와 teacher 크기가 같으므로 A와 B의 절감량이 같습니다. 측정 시간은 커널 효율 차이로 이와 조금 다를 수 있습니다.

---

## 코드 구조

```text
takd/
  data.py      COCO single 캐시(GPU 상주), 셔플·반전·정규화, 스모크용 합성 데이터
  models.py    DeiT(공식 파라미터 이름), 토큰 제거, CLS attention, 층별 특징, 폭 확장
  losses.py    CE / logit KD / PKT(information flow) + critical period
  flops.py     해석적 FLOPs
  engine.py    run 하나의 학습(재개, 시간·FLOPs 측정, 검증 선택, 진단, result.json)
  metrics.py   정확도, masking 진단
  plan.py      A–F 정의, run 의존 관계와 실행 순서
  report.py    CSV·SUMMARY.md 생성, figures.py 그림
  cli.py       prepare / plan / run / status / report / smoke
configs/hkd.json, configs/logit.json   두 suite의 모든 하이퍼파라미터
datasets/coco_single/prepare.py, scripts/check_assets.py   기준 레포에서 가져온 데이터 준비·검증
scripts/*.sh   서버용 명령
tests/         단위 테스트 (FLOPs 식, 폭 확장의 함수 보존, CLS attention, 손실, 계획)
```

각 run은 설정·데이터 해시·teacher 체크포인트 해시·학습 코드 해시로 서명됩니다. 다시 실행할 때 서명이 같은 완료 run은 건너뛰고, 다르면 멈춥니다(해당 폴더를 옮기거나 `--allow-code-change`).

## 참고문헌
- N. Passalis, M. Tzelepi, A. Tefas. *Heterogeneous Knowledge Distillation using Information Flow Modeling.* CVPR 2020. [arXiv:2005.00727](https://arxiv.org/abs/2005.00727), 코드 [passalis/pkth](https://github.com/passalis/pkth)
- S. Son, J. Ryu, N. Lee, J. Lee. *The Role of Masking for Efficient Supervised Knowledge Distillation of Vision Transformers.* ECCV 2024. [arXiv:2302.10494](https://arxiv.org/abs/2302.10494), 코드 [effl-lab/MaskedKD](https://github.com/effl-lab/MaskedKD)
- S. I. Mirzadeh et al. *Improved Knowledge Distillation via Teacher Assistant.* AAAI 2020.
- H. Touvron et al. *Training data-efficient image transformers & distillation through attention.* ICML 2021.
- T. Chen, I. Goodfellow, J. Shlens. *Net2Net: Accelerating Learning via Knowledge Transfer.* ICLR 2016.
- 기준 벤치마크: [jeehoo0507/kshs-aimlab-benchmarks](https://github.com/jeehoo0507/kshs-aimlab-benchmarks)
