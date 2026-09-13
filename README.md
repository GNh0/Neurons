# Neurons · CNS Observatory

MaleCNS v1.0의 실제 초파리 중추신경계 연결 데이터를 불러오고, 단순화한 LIF 계산과 3D 지도를 관찰하는 Windows 로컬 앱입니다.

| 계산에 포함된 데이터 | 개수 |
| --- | ---: |
| 분류된 뉴런 | 166,700 |
| 방향성 뉴런 쌍 | 25,582,938 |
| 쌍별 가중치로 보존된 시냅스 접촉 | 124,177,617 |
| 위치가 제공된 뉴런 | 140,638 |

전체 계산 그래프는 표본화하지 않습니다. 1.24억 접촉은 뉴런 쌍별 정수 가중치로 집계됩니다. 개별 접촉의 생화학 상태를 1.24억 개 객체로 계산하는 모델은 아닙니다.

## Windows 배포본

[GitHub Releases](https://github.com/GNh0/Neurons/releases)의 **Neurons-Windows-x64.zip**을 압축 해제하고 **Run Neurons.cmd**를 실행합니다. Python 실행 환경, 라이브러리, 원본 데이터와 컴파일된 연결망이 포함됩니다. LLM 모델과 Ollama는 포함하지 않습니다.

모니터링과 뉴런 자극 실험은 LLM 없이 실행됩니다. 대화에는 [Ollama](https://ollama.com/download/windows)를 설치하고 로컬 모델을 받으세요.

`web/index.html`을 직접 여는 것은 앱 실행이 아닙니다. 이 경우 실행 안내 화면이 표시됩니다. 반드시 Run Neurons.cmd 또는 start.ps1을 사용하세요.

```powershell
ollama pull qwen3.5:4b
```

Ollama 실행 후 앱의 **모듈 → 연결 다시 확인**을 누릅니다. 설치된 모델을 선택할 수 있으며 표준 포트 11434와 개발용 포트 11435를 확인합니다.

## 소스에서 실행

Python 3.10 이상이 설치된 Windows PowerShell:

```powershell
git clone https://github.com/GNh0/Neurons.git
cd Neurons
.\setup.ps1
.\start.ps1 -OpenBrowser
```

설치 스크립트가 전용 가상 환경을 만들고 공식 서버에서 약 1.11 GB의 원본 데이터를 받아 무결성을 확인한 뒤 연결망을 구성합니다. 소스 저장소에는 대용량 데이터와 실행 환경이 포함되지 않습니다. 기본 주소는 http://127.0.0.1:8878 입니다. `-Port`로 변경할 수 있습니다. 이 버전은 RAM 32 GB의 Windows PC에서 확인했습니다.

## 관측과 대화

- 회전·확대·뉴런 선택이 가능한 WebGL 3D 지도.
- 막전위, 발화, 입력·출력 연결과 접촉 수 확인.
- 직접 자극, 실행·일시정지·10 ms 진행.
- 별도 SQLite 경험 기억, 중복 강화, 피드백과 참조 기억.
- 로컬 언어모델 대화 및 관측 가능한 실행 기록.

점은 세포체 또는 세포체 연결부입니다. 선은 뉴런 위치 사이의 집계 연결이며 개별 시냅스의 실제 접촉 좌표가 아닙니다. 전체 보기는 표시용 선 16,000개, 선택 뉴런은 입력·출력 각각 최대 180개를 사용합니다. 좌표 미제공 뉴런 26,062개도 계산에 포함되며 선택적으로 비해부학적 위치에 따로 표시합니다.

## 범위

LIF는 단일 구획·선형 가중치·공통 지연의 계산 모델입니다. 전달물질별 부호는 가정이고 조절성·미상은 빠른 전류 효과 0입니다. 대화는 별도 LLM이 생성합니다. 기억을 저장해도 원본 뇌 연결 가중치는 바뀌지 않습니다. 이 보존 버전에는 자율 인터넷 연구 루프가 없습니다. 의식, 사람 같은 사고, 생물학적 학습·진화가 입증된 시스템이 아닙니다.

기억·모듈 설정·대화는 `.data/memory.sqlite3`에 저장됩니다. 막전위와 누적 발화는 재시작하면 초기화됩니다. 관측 내보내기는 현재 상태, 최신 기억 최대 250개와 최근 대화 40개이며 전체 시뮬레이터 재개용 스냅샷이 아닙니다. 화면 FPS와 생물학적 시간의 계산 속도는 다릅니다.

## 검증

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\validate_full_graph.py
```

[검증 기록](docs/validation.md) · [구조 설명](docs/architecture.md)

## 출처

[MaleCNS 공식 데이터](https://male-cns.janelia.org/download/) v1.0 공개일은 2026-06-08, [Cell 논문](https://doi.org/10.1016/j.cell.2026.08.015) 발표일은 2026-09-03입니다. 데이터 CC BY 4.0 조건과 원저자 출처를 유지합니다. 변환 내용은 [NOTICE](NOTICE.md)에 기록했습니다.
