from math import ceil
from os.path import join
import sys
import numpy as np
import time
from datetime import datetime

## hyperparams
# min_ps=11	# weakest clade mut's per_sum=11.52 (next 26.87)
# maxeps=1000
# eps_scaler_const=1.5
# exp_dim=30.	# the eps_scaler diminishing effect (higher the exp_dim the lesser the es penalized)
# min_clength=10	# minimum length of cluster

from src.utils import mutation_filtering

POS = 'Position'
FREQ = 'Frequency'
PER = 'Percentage'
ENT = 'Entropy'

HSCORE = 'H-score'
HSCORE_SUM = 'H-score_sum'
HSCORE_AVR = 'H-score_avr'

PER_SUM = 'per_sum'
ENT_SUM = 'ent_sum'
PER_AVR = 'per_avr'
ENT_AVR = 'ent_avr'

MAX_DEPS = 50
MAX_EPS = 1000
MIN_EPS= 10

EPSILON = 5
EPSILON_SCALING_FACTOR = 10

DIMINISHING_FACTOR = 3

MIN_CLUSTER_LENGTH = 10

CCM_MIN_FREQUENCY = 1000
CCM_MIN_PERCENTAGE = 0.01
CCM_MIN_ENTROPY = 0.45

CCM_MIN_PERCENTAGE_SUM = 0.01
CCM_MIN_ENTROPY_SUM = 0.05

CCM_MIN_PERCENTAGE_AVR = 0.001
CCM_MIN_ENTROPY_AVR = 0.01

CCM_MIN_HSCORE_SUM = 0.05
CCM_MIN_HSCORE_AVR = 0.01
CCM_MIN_HSCORE = 0.03

MIN_MUTATIONS = 5
MIN_CLUSTERS = 5

# ============================================================
# 로깅 유틸리티
# ============================================================
class Logger:
	"""상세한 로그 출력을 위한 유틸리티 클래스"""

	VERBOSE = True  # 상세 로그 출력 여부

	@staticmethod
	def header(title, char='='):
		"""섹션 헤더 출력"""
		line = char * 60
		print(f"\n{line}")
		print(f"  {title}")
		print(f"{line}")

	@staticmethod
	def subheader(title, char='-'):
		"""서브 섹션 헤더 출력"""
		line = char * 40
		print(f"\n{line}")
		print(f"  {title}")
		print(f"{line}")

	@staticmethod
	def step(step_num, description):
		"""단계 정보 출력"""
		print(f"\n[STEP {step_num}] {description}")
		print("-" * 50)

	@staticmethod
	def info(message, indent=0):
		"""일반 정보 출력"""
		prefix = "  " * indent
		print(f"{prefix}[INFO] {message}")

	@staticmethod
	def detail(message, indent=1):
		"""상세 정보 출력"""
		if Logger.VERBOSE:
			prefix = "  " * indent
			print(f"{prefix}├─ {message}")

	@staticmethod
	def stat(label, value, indent=1):
		"""통계 정보 출력"""
		prefix = "  " * indent
		print(f"{prefix}│ {label}: {value}")

	@staticmethod
	def progress(current, total, prefix='Progress', suffix=''):
		"""진행률 표시"""
		percent = 100 * (current / float(total))
		bar_length = 30
		filled = int(bar_length * current // total)
		bar = '█' * filled + '░' * (bar_length - filled)
		sys.stdout.write(f'\r  {prefix} |{bar}| {percent:.1f}% ({current}/{total}) {suffix}')
		if current == total:
			print()

	@staticmethod
	def success(message):
		"""성공 메시지 출력"""
		print(f"  [OK] {message}")

	@staticmethod
	def warning(message):
		"""경고 메시지 출력"""
		print(f"  [WARN] {message}")

	@staticmethod
	def summary(title, stats_dict):
		"""요약 통계 출력"""
		print(f"\n  [{title}]")
		for key, value in stats_dict.items():
			if isinstance(value, float):
				print(f"    • {key}: {value:.6f}")
			else:
				print(f"    • {key}: {value}")

	@staticmethod
	def time_elapsed(start_time, operation_name):
		"""경과 시간 출력"""
		elapsed = time.time() - start_time
		print(f"  [TIME] {operation_name}: {elapsed:.2f}초")
		return elapsed


class Cluster(object):
	def __init__(self, cid, idx, f, af):
		self.cid=cid
		self.cpts=[idx]
		self.totf=[f] 	# total num of mut freq in cluster
		self.af=af 		# average mut freq in cluster
		self.length=0 	# length of cluster
		self.size=1		# num of muts in cluster
		self.seq=''		# sequence of cluster

	def add_cpt(self, idx, f):
		self.cpts.append(idx)
		self.totf.append(f)
		self.af=np.mean(self.totf)	# update average mut freq of cluster
		self.length=self.cpts[-1]-self.cpts[0]	# updated cluster length(bp)
		self.size+=1				# update cluster size

	def __str__(self):
		return '%d\t%d:%d\t%d\t%d\t%.3f\t%s'%(self.cid, self.cpts[0], self.cpts[-1], self.length, self.size, self.af, self.seq)


def init(d, info):
	Logger.header('MUTCLUST 초기화')

	print(f"\n  [입력 데이터]")
	print(f"    • 파일: '{info.fin}'")
	print(f"    • 크기: {d.shape[0]} positions × {d.shape[1]} columns")

	print(f"\n  [출력 디렉토리]")
	print(f"    • 경로: '{info.outdir}'")

	print(f"\n  [하이퍼파라미터]")
	print(f"    ┌{'─'*40}┐")
	print(f"    │ {'Parameter':<25} {'Value':>12} │")
	print(f"    ├{'─'*40}┤")
	print(f"    │ {'Min Epsilon':<25} {info.eps:>12} │")
	print(f"    │ {'Max Epsilon':<25} {info.maxeps:>12} │")
	print(f"    │ {'Min per_sum':<25} {info.min_persum:>12.2f} │")
	print(f"    │ {'Eps Scaling Factor':<25} {info.eps_scaler_const:>12.1f} │")
	print(f"    │ {'Diminishing Factor':<25} {info.es_control_const:>12} │")
	print(f"    │ {'Min Cluster Length':<25} {info.min_cluster_length:>12} │")
	print(f"    └{'─'*40}┘")

	print(f"\n  [CCM 선정 기준]")
	print(f"    • Min H-score: {CCM_MIN_HSCORE}")
	print(f"    • Min H-score Sum: {CCM_MIN_HSCORE_SUM}")
	print(f"    • Min H-score Avg: {CCM_MIN_HSCORE_AVR}")
	print(f"    • Min Mutations: {MIN_MUTATIONS}")


class get_eps_stats(object):
	def __init__(self, idx, pos, df, lr_index, lr_distance, es):
		self.idx=idx
		self.i=pos
		self.i_per=df.loc[idx, PER]
		self.i_freq=df.loc[idx, FREQ]
		self.i_ent=df.loc[idx, ENT]
		self.i_hscore = df.loc[idx, HSCORE]

		self.l_dist= lr_distance[0]
		self.r_dist = lr_distance[1]
		ccm_df = df.loc[lr_index[0]:lr_index[1] + 1, :]

		self.length = len(ccm_df)
		self.l_pos = df.loc[lr_index[0], POS]
		self.r_pos = df.loc[lr_index[1], POS]

		self.mut_n = len(ccm_df[ccm_df[HSCORE]>0])
		self.eps_scaler = es

		self.freq_sum= ccm_df[FREQ].sum()
		self.freq_avr=self.freq_sum/self.length

		self.per_sum= ccm_df[PER].sum()
		self.per_avr=self.per_sum/self.length

		self.ent_sum= ccm_df[ENT].sum()
		self.ent_avr=self.ent_sum/self.length

		self.hscore_sum=ccm_df[HSCORE].sum()
		self.hscore_avr=self.hscore_sum/self.length

	def to_list(self):
		param_list=[self.idx, self.i, self.i_freq, self.i_per, self.length, self.freq_sum, self.freq_avr, self.per_sum, self.per_avr, self.ent_sum, self.ent_avr, self.eps_scaler, self.l_dist, self.r_dist]
		return param_list

	def to_dict(self):
		param_dict={'index':self.idx, POS:self.i, FREQ:self.i_freq, PER:self.i_per,ENT:self.i_ent,HSCORE:self.i_hscore, 'length':self.length, 'freq_sum':self.freq_sum, 'freq_avr':self.freq_avr,
					PER_SUM:self.per_sum, PER_AVR:self.per_avr, ENT_SUM:self.ent_sum, ENT_AVR:self.ent_avr, HSCORE_SUM:self.hscore_sum, HSCORE_AVR:self.hscore_avr,
					'eps_scaler':self.eps_scaler, 'left_distance':self.l_dist, 'right_distance':self.r_dist, 'l_pos':self.l_pos, 'r_pos':self.r_pos, 'mut_n':self.mut_n}
		return param_dict


def get_eps_region(mutclust_input_df, cur_index, info):
	dist_l = dist_r = 0	# left and right distance from i
	cur_l_index =  cur_index - 1
	cur_r_index = cur_index + 1
	cur_hscore = mutclust_input_df.loc[cur_index, HSCORE]

	eps_scaler = ceil(EPSILON_SCALING_FACTOR * cur_hscore)
	ldeps = rdeps = eps_scaler*EPSILON

	if cur_l_index < 0:
		cur_l_index = cur_index
	if cur_r_index >= mutclust_input_df.shape[0]-1:
		cur_r_index = cur_index

	# search left boundary
	if ldeps > info.maxeps:
		ldeps = info.maxeps # max at maxeps

	while dist_l < ldeps and cur_l_index >= 0:
		ld = mutclust_input_df.loc[cur_index, POS] - mutclust_input_df.loc[cur_l_index, POS]
		if ld > ldeps:
			break
		cur_l_index -= 1
		dist_l = ld
	cur_l_index += 1

	# search right boundary
	if rdeps > info.maxeps:
		rdeps = info.maxeps	# max at maxeps

	while dist_r < rdeps and cur_r_index < mutclust_input_df.shape[0]:
		# print(rdeps)
		rd = mutclust_input_df.loc[cur_r_index, POS] - mutclust_input_df.loc[cur_index, POS]
		if rd > rdeps:
			break
		cur_r_index += 1
		dist_r = rd
	cur_r_index -= 1

	return [cur_l_index, cur_r_index], [cur_index-cur_l_index, cur_r_index-cur_index], eps_scaler


def expand_cluster(ccm_idx, total_mutation_info_list, info, verbose=False):
	"""클러스터 확장 - 양방향으로 CCM에서 확장"""
	left_cur_dist = right_cur_dist = 0			# left and right distance from pos
	left_cur_index = ccm_idx - 1				# left moving index
	right_cur_index = ccm_idx + 1				# right moving index
	mut_n = len(total_mutation_info_list)
	if right_cur_index >= mut_n:
		right_cur_index = ccm_idx

	es_l = es_r = total_mutation_info_list[ccm_idx]['eps_scaler']		# ccm's pre-computed eps scaler (es)(per/1.5)
	left_max_dist = total_mutation_info_list[ccm_idx]['left_distance']
	right_max_dist = total_mutation_info_list[ccm_idx]['right_distance']		# ccm's pre-computed deps length (dist)

	initial_left_max = left_max_dist
	initial_right_max = right_max_dist
	left_expansion_steps = 0
	right_expansion_steps = 0

	# expand left
	while left_cur_dist<left_max_dist and left_cur_index>=0:
		ld = total_mutation_info_list[ccm_idx][POS] - total_mutation_info_list[left_cur_index][POS]
		if ld > left_max_dist:
			break
		left_cur_dist = ld

		# decrease deps in respect to es of cur_l
		delta_es = es_l - total_mutation_info_list[left_cur_index]['eps_scaler']	# delta(eps)=es_cur-es_ccm
		es_l = es_l - (delta_es) / info.es_control_const# diminish es by delta(eps)/exp_dim delta/30
		mut_deps = info.eps * es_l

		if mut_deps > 0:
			left_max_dist = mut_deps
		else:
			break
		left_cur_index -= 1
		left_expansion_steps += 1

	# expand right
	while right_cur_dist<right_max_dist and right_cur_index < mut_n:
		rd = total_mutation_info_list[right_cur_index][POS] - total_mutation_info_list[ccm_idx][POS]
		if rd>right_max_dist:
			break
		right_cur_dist=rd
		# decrease deps in respect to es of cur_r
		delta_es = es_r - total_mutation_info_list[right_cur_index]['eps_scaler']	# delta(eps)=eps_i-eps_curl
		es_r = es_r - (delta_es)/info.es_control_const	# diminish es by delta(eps)/exp_dim (default: 30)
		mut_deps = info.eps * es_r

		if mut_deps > 0:
			right_max_dist = mut_deps
		else:
			break
		right_cur_index+=1
		right_expansion_steps += 1

	if right_cur_index == mut_n:
		right_cur_index-=1
	if left_cur_index < 0:
		left_cur_index = 0

	ret_dict = { 'length': total_mutation_info_list[right_cur_index][POS] - total_mutation_info_list[left_cur_index][POS] + 1,
				'ccm_position':ccm_idx,
				'mut_positions': sorted([a[POS] for a in total_mutation_info_list[left_cur_index:right_cur_index+1] if a[HSCORE] > 0])}
	ret_dict['left_position'] = ret_dict['mut_positions'][0]
	ret_dict['right_position'] = ret_dict['mut_positions'][-1]

	# 상세 로그 (verbose 모드)
	if verbose:
		Logger.detail(f"CCM Position: {total_mutation_info_list[ccm_idx][POS]}")
		Logger.detail(f"Initial eps_scaler: {total_mutation_info_list[ccm_idx]['eps_scaler']}")
		Logger.detail(f"Left expansion: {left_expansion_steps} steps, dist: {initial_left_max:.1f} → {left_max_dist:.1f}")
		Logger.detail(f"Right expansion: {right_expansion_steps} steps, dist: {initial_right_max:.1f} → {right_max_dist:.1f}")
		Logger.detail(f"Result: [{ret_dict['left_position']} - {ret_dict['right_position']}], {len(ret_dict['mut_positions'])} mutations")

	return ret_dict


def dynaclust(total_mutation_info_list, ccm_index_list, info, tag, i):
	"""동적 클러스터링 및 병합"""
	Logger.header('PHASE 2: 동적 클러스터링')
	start_time = time.time()

	# ========================================
	# STEP 2.1: 클러스터 확장
	# ========================================
	Logger.step("2.1", "CCM 기반 클러스터 확장")
	Logger.info(f"총 {len(ccm_index_list)}개 CCM에 대해 클러스터 확장 시작")

	cluster_list=[]
	expansion_stats = {
		'total_mutations': 0,
		'min_length': float('inf'),
		'max_length': 0,
		'total_length': 0
	}

	sample_indices = [0, len(ccm_index_list)//4, len(ccm_index_list)//2,
					  3*len(ccm_index_list)//4, len(ccm_index_list)-1]

	for idx, ccm_idx in enumerate(ccm_index_list):
		# 일부 샘플에 대해 상세 로그 출력
		verbose = (idx in sample_indices[:3]) if len(ccm_index_list) > 10 else True

		ret_dict = expand_cluster(ccm_idx, total_mutation_info_list, info, verbose=verbose)
		cluster_list.append(ret_dict)

		# 통계 수집
		cluster_len = ret_dict['length']
		mut_count = len(ret_dict['mut_positions'])
		expansion_stats['total_mutations'] += mut_count
		expansion_stats['min_length'] = min(expansion_stats['min_length'], cluster_len)
		expansion_stats['max_length'] = max(expansion_stats['max_length'], cluster_len)
		expansion_stats['total_length'] += cluster_len

		# 진행률 표시
		if (idx + 1) % 100 == 0 or idx == len(ccm_index_list) - 1:
			Logger.progress(idx + 1, len(ccm_index_list),
						   prefix='클러스터 확장',
						   suffix=f'현재: {len(cluster_list)} clusters')

	print()
	Logger.summary("클러스터 확장 통계", {
		"생성된 클러스터 수": len(cluster_list),
		"총 변이 수": expansion_stats['total_mutations'],
		"최소 길이": expansion_stats['min_length'],
		"최대 길이": expansion_stats['max_length'],
		"평균 길이": expansion_stats['total_length'] / len(cluster_list)
	})
	Logger.time_elapsed(start_time, "클러스터 확장")

	# ========================================
	# STEP 2.2: 클러스터 병합
	# ========================================
	merge_start = time.time()
	Logger.step("2.2", "겹치는 클러스터 병합")

	merged_clusters=[]
	cluster_list = sorted(cluster_list, key=lambda x: x['left_position'])
	clst_n = len(cluster_list)

	Logger.info(f"정렬 완료: {clst_n}개 클러스터를 위치 순으로 정렬")
	Logger.info(f"병합 조건: right_pos(i) >= left_pos(j) → merge")
	Logger.info(f"최소 변이 수: {MIN_MUTATIONS}개 이상만 유지")

	merge_count = 0
	skip_count = 0

	i = 0
	while i < clst_n:
		lpos=cluster_list[i]['left_position']
		rpos=cluster_list[i]['right_position']
		mut_list=cluster_list[i]['mut_positions']
		merged_in_this_round = 0
		j = i+1
		while j < clst_n:
			if rpos < cluster_list[j]['left_position']: # cluster not in cluster next
				i = j
				break
			else: # cluster in cluster
				lpos = min(lpos, cluster_list[j]['left_position'])
				rpos = max(rpos, cluster_list[j]['right_position'])
				mut_list.extend(cluster_list[j]['mut_positions'])
				mut_list = [a for a in set(mut_list)]
				mut_list.sort()
				merged_in_this_round += 1
				j+=1
				if j >= clst_n:
					i=j
					break

		if j>= clst_n:
			i+=1

		if len(mut_list) >= MIN_MUTATIONS: # saving merged cluster
			ret_dict = { 'left_position':lpos,
						'right_position':rpos,
							'length': rpos-lpos+1,
						'mut_positions':','.join([str(a) for a in mut_list])}
			merged_clusters.append(ret_dict)
			if merged_in_this_round > 0:
				merge_count += merged_in_this_round
		else:
			skip_count += 1

	Logger.info(f"병합 결과:")
	Logger.detail(f"병합된 클러스터 쌍 수: {merge_count}")
	Logger.detail(f"제외된 클러스터 (변이 < {MIN_MUTATIONS}): {skip_count}개")
	Logger.detail(f"최종 클러스터 수: {len(merged_clusters)}개")

	# 최종 클러스터 통계
	if merged_clusters:
		lengths = [c['length'] for c in merged_clusters]
		mut_counts = [len(c['mut_positions'].split(',')) for c in merged_clusters]

		Logger.summary("최종 클러스터 통계", {
			"총 클러스터 수": len(merged_clusters),
			"평균 길이 (bp)": np.mean(lengths),
			"최소 길이": min(lengths),
			"최대 길이": max(lengths),
			"평균 변이 수": np.mean(mut_counts),
			"최소 변이 수": min(mut_counts),
			"최대 변이 수": max(mut_counts)
		})

	Logger.time_elapsed(merge_start, "클러스터 병합")

	# ========================================
	# STEP 2.3: 결과 저장
	# ========================================
	Logger.step("2.3", "결과 파일 저장")

	output_file = '%s/clusters_%s.txt'%(info.outdir, str(tag))
	cl_outf=open(output_file, 'w')
	header=merged_clusters[0].keys()
	cl_outf.write('%s\n'%('\t'.join(header)))
	for cluster_info_dict in merged_clusters:
		cl_outf.write('%s\n'%('\t'.join([str(x) for x in cluster_info_dict.values()])))
	cl_outf.close()

	Logger.success(f"클러스터 결과 저장: {output_file}")
	Logger.info(f"저장된 클러스터: {len(merged_clusters)}개")

	# 상위 5개 클러스터 미리보기
	print("\n  [상위 5개 클러스터 미리보기]")
	print(f"    {'Left':>8} {'Right':>8} {'Length':>8} {'Mutations':>10}")
	print(f"    {'-'*38}")
	for c in sorted(merged_clusters, key=lambda x: -len(x['mut_positions'].split(',')))[:5]:
		mut_count = len(c['mut_positions'].split(','))
		print(f"    {c['left_position']:>8} {c['right_position']:>8} {c['length']:>8} {mut_count:>10}")

	total_time = time.time() - start_time
	Logger.header(f'PHASE 2 완료 (총 {total_time:.2f}초)')

	return merged_clusters


def get_candidate_core_mutations(mutclust_input_df, info, tag, i):
	"""CCM (Candidate Core Mutation) 선별"""
	Logger.header('PHASE 1: CCM (Candidate Core Mutation) 선별')
	start_time = time.time()

	total_mutInfo_list=[]
	ccm_index_list=[]
	total_index_list=[]

	# ========================================
	# STEP 1.1: 변이 필터링
	# ========================================
	Logger.step("1.1", "유효 범위 필터링")
	Logger.info(f"원본 데이터: {len(mutclust_input_df)} positions")
	Logger.info(f"필터링 조건: 266 <= Position <= 29674 (유효 코딩 영역)")

	filtered_mutclust_input_df = mutation_filtering(mutclust_input_df)

	filtered_count = len(filtered_mutclust_input_df)
	removed_count = len(mutclust_input_df) - filtered_count

	Logger.success(f"필터링 완료")
	Logger.detail(f"유지된 positions: {filtered_count}")
	Logger.detail(f"제외된 positions: {removed_count}")
	Logger.detail(f"제외 비율: {removed_count/len(mutclust_input_df)*100:.2f}%")

	# 데이터 통계
	hscore_data = filtered_mutclust_input_df[HSCORE]
	Logger.summary("필터링 후 H-score 통계", {
		"평균": hscore_data.mean(),
		"표준편차": hscore_data.std(),
		"최소값": hscore_data.min(),
		"최대값": hscore_data.max(),
		"중앙값": hscore_data.median()
	})

	# ========================================
	# STEP 1.2: Epsilon 영역 계산 및 통계 수집
	# ========================================
	Logger.step("1.2", "각 변이의 Epsilon 영역 계산")
	Logger.info(f"Epsilon 계산 공식: eps_scaler = ceil({EPSILON_SCALING_FACTOR} × H-score)")
	Logger.info(f"탐색 범위: eps_scaler × {EPSILON} (최대: {info.maxeps})")

	rejected_stats = {
		'low_mut_n': 0,
		'low_hscore_sum': 0,
		'low_hscore_avr': 0,
		'low_hscore': 0
	}

	eps_scalers = []
	region_lengths = []

	for index, pos in enumerate(filtered_mutclust_input_df[POS]):
		lr_index, lr_distance, eps_scaler = get_eps_region(filtered_mutclust_input_df, index, info)
		mut_info = get_eps_stats(index, pos, filtered_mutclust_input_df, lr_index, lr_distance, eps_scaler)
		total_mutInfo_list.append(mut_info.to_dict())
		total_index_list.append(index)

		eps_scalers.append(eps_scaler)
		region_lengths.append(mut_info.length)

		# CCM 선정 조건 확인 (순차적 필터링)
		if mut_info.mut_n < MIN_MUTATIONS:
			rejected_stats['low_mut_n'] += 1
			continue
		if mut_info.hscore_sum < CCM_MIN_HSCORE_SUM:
			rejected_stats['low_hscore_sum'] += 1
			continue
		if mut_info.hscore_avr < CCM_MIN_HSCORE_AVR:
			rejected_stats['low_hscore_avr'] += 1
			continue
		if mut_info.i_hscore < CCM_MIN_HSCORE:
			rejected_stats['low_hscore'] += 1
			continue

		ccm_index_list.append(index)

		# 진행률 표시
		if (index + 1) % 5000 == 0:
			Logger.progress(index + 1, len(filtered_mutclust_input_df[POS]),
						   prefix='Epsilon 계산',
						   suffix=f'CCM: {len(ccm_index_list)}개')

	Logger.progress(len(filtered_mutclust_input_df[POS]), len(filtered_mutclust_input_df[POS]),
				   prefix='Epsilon 계산',
				   suffix=f'CCM: {len(ccm_index_list)}개')

	Logger.summary("Epsilon 영역 통계", {
		"평균 eps_scaler": np.mean(eps_scalers),
		"최대 eps_scaler": max(eps_scalers),
		"평균 영역 길이": np.mean(region_lengths),
		"최대 영역 길이": max(region_lengths)
	})

	# ========================================
	# STEP 1.3: CCM 선정 결과
	# ========================================
	Logger.step("1.3", "CCM 선정 결과")

	total_rejected = sum(rejected_stats.values())

	Logger.info(f"CCM 선정 기준 적용 결과:")
	Logger.detail(f"변이 수 < {MIN_MUTATIONS}로 제외: {rejected_stats['low_mut_n']}개")
	Logger.detail(f"H-score 합 < {CCM_MIN_HSCORE_SUM}로 제외: {rejected_stats['low_hscore_sum']}개")
	Logger.detail(f"H-score 평균 < {CCM_MIN_HSCORE_AVR}로 제외: {rejected_stats['low_hscore_avr']}개")
	Logger.detail(f"H-score < {CCM_MIN_HSCORE}로 제외: {rejected_stats['low_hscore']}개")

	print()
	Logger.success(f"최종 CCM 선정: {len(ccm_index_list)}개 / {filtered_count}개")
	Logger.detail(f"CCM 비율: {len(ccm_index_list)/filtered_count*100:.2f}%")

	# CCM 통계
	if ccm_index_list:
		ccm_hscores = [total_mutInfo_list[idx][HSCORE] for idx in ccm_index_list]
		ccm_positions = [total_mutInfo_list[idx][POS] for idx in ccm_index_list]

		Logger.summary("CCM H-score 통계", {
			"평균": np.mean(ccm_hscores),
			"표준편차": np.std(ccm_hscores),
			"최소값": min(ccm_hscores),
			"최대값": max(ccm_hscores)
		})

		# 상위 10개 CCM 출력
		print("\n  [상위 10개 CCM (H-score 기준)]")
		print(f"    {'Position':>10} {'H-score':>12} {'Freq':>10} {'mut_n':>8}")
		print(f"    {'-'*44}")

		sorted_ccm = sorted([(idx, total_mutInfo_list[idx]) for idx in ccm_index_list],
						   key=lambda x: -x[1][HSCORE])[:10]
		for idx, info_dict in sorted_ccm:
			print(f"    {info_dict[POS]:>10} {info_dict[HSCORE]:>12.6f} {info_dict[FREQ]:>10} {info_dict['mut_n']:>8}")

	# ========================================
	# STEP 1.4: 결과 파일 저장
	# ========================================
	Logger.step("1.4", "결과 파일 저장")

	# 전체 결과 저장
	total_output = '%s/total_results_%s.tsv'%(info.outdir, tag)
	with open(total_output, 'w') as outf:
		header='\t'.join(total_mutInfo_list[0].keys()) + '\n'
		outf.write(header)
		for index in total_index_list:
			outf.write('%s\n'%('\t'.join([str(x) for x in total_mutInfo_list[index].values()])))
	Logger.success(f"전체 변이 정보 저장: {total_output}")
	Logger.detail(f"저장된 레코드: {len(total_index_list)}개")

	# CCM 결과 저장
	ccm_output = '%s/ccm_%s.tsv'%(info.outdir, tag)
	with open(ccm_output, 'w') as outf:
		header='\t'.join(total_mutInfo_list[0].keys()) + '\n'
		outf.write(header)
		for index in ccm_index_list:
			outf.write('%s\n'%('\t'.join([str(x) for x in total_mutInfo_list[index].values()])))
	Logger.success(f"CCM 결과 저장: {ccm_output}")
	Logger.detail(f"저장된 CCM: {len(ccm_index_list)}개")

	total_time = time.time() - start_time
	Logger.header(f'PHASE 1 완료 (총 {total_time:.2f}초)')

	total_mutInfo_list = np.asarray(total_mutInfo_list)
	return total_mutInfo_list, ccm_index_list
