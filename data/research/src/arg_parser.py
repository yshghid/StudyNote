import argparse
import shutil
from os.path import join, exists, isdir, dirname
from os import mkdir
from datetime import datetime

from src.mlib import DIMINISHING_FACTOR, EPSILON, EPSILON_SCALING_FACTOR, MAX_EPS, MIN_CLUSTER_LENGTH, CCM_MIN_PERCENTAGE_SUM


class args_info:
	args = {}
	fin=str()
	ref=str()
	outdir=str()

	# hyper parameters
	eps=int()
	maxeps=int()
	min_persum=int()	# weakest clade mut's per_sum=11.52 (next 26.87)
	eps_scaler_const=float()
	es_control_const=float()	# the eps_scaler diminishing effect (higher the exp_dim the lesser the es penalized)
	min_cluster_length=int()	# minimum length of cluster


def print_banner():
	"""프로그램 배너 출력"""
	banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ███╗   ███╗██╗   ██╗████████╗ ██████╗██╗     ██╗   ██╗   ║
║   ████╗ ████║██║   ██║╚══██╔══╝██╔════╝██║     ██║   ██║   ║
║   ██╔████╔██║██║   ██║   ██║   ██║     ██║     ██║   ██║   ║
║   ██║╚██╔╝██║██║   ██║   ██║   ██║     ██║     ██║   ██║   ║
║   ██║ ╚═╝ ██║╚██████╔╝   ██║   ╚██████╗███████╗╚██████╔╝   ║
║   ╚═╝     ╚═╝ ╚═════╝    ╚═╝    ╚═════╝╚══════╝ ╚═════╝    ║
║                                                              ║
║         Mutation Hotspot Clustering Algorithm                ║
║                     Version 1.0                              ║
╚══════════════════════════════════════════════════════════════╝
"""
	print(banner)
	print(f"  실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
	print()


def set_env(input=None, reference=None, output=None, verbose=False):
	"""
	환경 설정 및 파라미터 파싱

	Args:
		input: 입력 파일 경로
		reference: 참조 게놈 경로
		output: 출력 디렉토리 경로
		verbose: 상세 로그 출력 여부

	Returns:
		args_info: 설정 정보 객체
	"""
	if verbose:
		print_banner()

	info = args_info()
	parser = argparse.ArgumentParser(
		prog="cluster.py",
		description="MutClust: Mutation Hotspot Clustering Algorithm",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  python cluster.py -f data/input.csv -o results/
  python cluster.py -f data/input.csv -e 10 -maxeps 500 -es 15
		"""
	)

	# required arguements
	parser.add_argument('-f', '--input_file', type=str,
						default='/data3/projects/2020_MUTCLUST/Data/Rawdata/COVID19/nucleotide_data/mutclust_input_data.txt',
						help='mutation frequency data file', required=False)
	parser.add_argument('-r', '--ref', type=str,
						default='/data3/projects/2020_MUTCLUST/Data/Rawdata/COVID19/nucleotide_data/new_reference.fasta',
						help='the reference genome', required=False)

	# parameters
	parser.add_argument('-e', '--eps', type=int, default=EPSILON,
						help='width of window (epsilon)', required=False)
	parser.add_argument('-maxeps', '--maxeps', type=int, default=MAX_EPS,
						help='maximum eps', required=False)
	parser.add_argument('-minps', '--min_ps', type=int, default=CCM_MIN_PERCENTAGE_SUM,
						help='minimum per_sum', required=False)
	parser.add_argument('-es', '--eps_scaler_const', type=float, default=EPSILON_SCALING_FACTOR,
						help='eps scaling factor', required=False)
	parser.add_argument('-exd', '--exp_dim', type=float, default=DIMINISHING_FACTOR,
						help='cluster expansion es diminishing factor', required=False)
	parser.add_argument('-minl', '--min_clength', type=int, default=MIN_CLUSTER_LENGTH,
						help='minimum cluster length', required=False)

	args = vars(parser.parse_args())
	info.args = args

	if input == None:
		info.fin = args['input_file']
	else:
		info.fin = input

	if output == None:
		info.outdir = args['outdir'] if 'outdir' in args and args['outdir'] else 'output'
	else:
		info.outdir = output
		info.plot_outdir = output
		info.mutation_info_outdir = output

	# 입력 파일 검증
	if verbose:
		print("=" * 60)
		print("  환경 설정")
		print("=" * 60)

	if not exists(input if input else info.fin):
		print(f"\n  [ERROR] 입력 파일을 찾을 수 없습니다!")
		print(f"          경로: {input if input else info.fin}")
		print(f"\n  프로그램을 종료합니다.")
		exit(1)

	if verbose:
		print(f"\n  [파일 경로]")
		print(f"    ├─ 입력 파일: {info.fin}")
		if exists(info.fin):
			import os
			file_size = os.path.getsize(info.fin)
			print(f"    │  └─ 파일 크기: {file_size / 1024:.1f} KB")
		print(f"    └─ 출력 디렉토리: {info.outdir}")

		# 출력 디렉토리 생성/확인
		if not exists(info.outdir):
			mkdir(info.outdir)
			print(f"       └─ [생성됨]")
		else:
			print(f"       └─ [존재함]")

	# 파라미터 설정
	info.eps = args['eps']
	info.maxeps = args['maxeps']
	info.min_persum = args['min_ps']
	info.maxeps = args['maxeps']
	info.eps_scaler_const = args['eps_scaler_const']
	info.es_control_const = args['exp_dim']
	info.min_cluster_length = args['min_clength']

	if verbose:
		print(f"\n  [알고리즘 파라미터]")
		print(f"    ┌{'─'*50}┐")
		print(f"    │ {'파라미터':<28} {'값':>18} │")
		print(f"    ├{'─'*50}┤")
		print(f"    │ {'Epsilon (기본 탐색 폭)':<28} {info.eps:>18} │")
		print(f"    │ {'Max Epsilon (최대 탐색 범위)':<28} {info.maxeps:>18} │")
		print(f"    │ {'Min Percentage Sum':<28} {info.min_persum:>18} │")
		print(f"    │ {'Eps Scaling Factor':<28} {info.eps_scaler_const:>18.1f} │")
		print(f"    │ {'Diminishing Factor':<28} {info.es_control_const:>18.1f} │")
		print(f"    │ {'Min Cluster Length':<28} {info.min_cluster_length:>18} │")
		print(f"    └{'─'*50}┘")

		# 파라미터 설명
		print(f"\n  [파라미터 설명]")
		print(f"    • Epsilon: 기본 탐색 윈도우 크기 (bp)")
		print(f"    • Max Epsilon: Epsilon 영역의 최대 크기 제한")
		print(f"    • Eps Scaling Factor: H-score를 eps_scaler로 변환하는 계수")
		print(f"      └─ eps_scaler = ceil(H-score × {info.eps_scaler_const})")
		print(f"    • Diminishing Factor: 클러스터 확장 시 거리 감소 제어")
		print(f"      └─ 값이 클수록 확장이 느리게 감소")

		print(f"\n  [공식]")
		print(f"    • 탐색 범위 = eps_scaler × Epsilon")
		print(f"    • 확장 거리 감소 = delta_eps / {info.es_control_const}")

	return info


def validate_parameters(info):
	"""
	파라미터 유효성 검증

	Args:
		info: args_info 객체

	Returns:
		bool: 유효성 검사 통과 여부
	"""
	print(f"\n  [파라미터 검증]")

	errors = []
	warnings = []

	# Epsilon 검증
	if info.eps < 1:
		errors.append(f"Epsilon은 1 이상이어야 합니다 (현재: {info.eps})")
	if info.eps > 100:
		warnings.append(f"Epsilon이 큽니다 (현재: {info.eps}). 메모리 사용량이 증가할 수 있습니다.")

	# Max Epsilon 검증
	if info.maxeps < info.eps:
		errors.append(f"Max Epsilon은 Epsilon보다 커야 합니다 (maxeps: {info.maxeps}, eps: {info.eps})")
	if info.maxeps > 5000:
		warnings.append(f"Max Epsilon이 매우 큽니다 (현재: {info.maxeps})")

	# Scaling Factor 검증
	if info.eps_scaler_const <= 0:
		errors.append(f"Eps Scaling Factor는 양수여야 합니다 (현재: {info.eps_scaler_const})")

	# Diminishing Factor 검증
	if info.es_control_const <= 0:
		errors.append(f"Diminishing Factor는 양수여야 합니다 (현재: {info.es_control_const})")

	# 결과 출력
	if errors:
		print(f"    [ERROR] 파라미터 오류 발견:")
		for err in errors:
			print(f"      • {err}")
		return False

	if warnings:
		print(f"    [WARN] 경고:")
		for warn in warnings:
			print(f"      • {warn}")

	if not errors and not warnings:
		print(f"    [OK] 모든 파라미터가 유효합니다.")

	return True


def print_runtime_info():
	"""실행 환경 정보 출력"""
	import sys
	import platform

	print(f"\n  [실행 환경]")
	print(f"    • Python 버전: {sys.version.split()[0]}")
	print(f"    • 운영체제: {platform.system()} {platform.release()}")
	print(f"    • 프로세서: {platform.processor() or 'Unknown'}")

	try:
		import numpy as np
		print(f"    • NumPy 버전: {np.__version__}")
	except:
		pass

	try:
		import pandas as pd
		print(f"    • Pandas 버전: {pd.__version__}")
	except:
		pass
