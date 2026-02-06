from multiprocessing import Pool
import os
import base64
from os import  listdir,  walk
from os.path import isfile, join, isdir
import numpy as np
from scipy.stats import entropy
import pandas as pd
from pandas import read_csv, Series
import pickle
import time

import matplotlib.pyplot as plt
import matplotlib.transforms
import matplotlib.patches

HSCORE = 'H-score'
POS = 'Position'
FREQ = 'Frequency'
PER = 'Percentage'
ENT = 'Entropy'
NUCLEOTIDE_ANNOTATION_PATH = 'covid_annotation.tsv'


def readPickle(filepath):
    print(f"  [INFO] Pickle 파일 로드: {filepath}")
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    print(f"  [OK] 로드 완료")
    return data


def mutation_filtering(mutclust_input_df, verbose=True):
    """
    유효 범위 내 변이만 필터링 (266 <= Position <= 29674)

    Args:
        mutclust_input_df: 변이 데이터프레임
        verbose: 상세 로그 출력 여부

    Returns:
        필터링된 데이터프레임
    """
    start_time = time.time()

    original_count = len(mutclust_input_df)

    # 코딩 영역 필터링 (266: 첫 번째 단백질 시작, 29674: 마지막 단백질 끝)
    new_mutclust_input_df = mutclust_input_df.loc[
        (mutclust_input_df[POS] >= 266) & (mutclust_input_df[POS] <= 29674)
    ]
    new_mutclust_input_df.reset_index(drop=True, inplace=True)

    filtered_count = len(new_mutclust_input_df)
    removed_count = original_count - filtered_count

    if verbose:
        print(f"\n  ┌{'─'*50}┐")
        print(f"  │ {'변이 필터링 결과':^48} │")
        print(f"  ├{'─'*50}┤")
        print(f"  │ {'원본 positions:':<25} {original_count:>21} │")
        print(f"  │ {'필터링 후 positions:':<25} {filtered_count:>21} │")
        print(f"  │ {'제외된 positions:':<25} {removed_count:>21} │")
        print(f"  │ {'유지 비율:':<25} {filtered_count/original_count*100:>20.2f}% │")
        print(f"  └{'─'*50}┘")

        # 제외된 영역 상세
        if removed_count > 0:
            left_removed = len(mutclust_input_df[mutclust_input_df[POS] < 266])
            right_removed = len(mutclust_input_df[mutclust_input_df[POS] > 29674])
            print(f"\n  제외 상세:")
            print(f"    • 5' UTR (< 266): {left_removed}개")
            print(f"    • 3' UTR (> 29674): {right_removed}개")

    elapsed = time.time() - start_time
    if verbose:
        print(f"  [TIME] 필터링 소요 시간: {elapsed:.3f}초")

    return new_mutclust_input_df


def get_GeneInfo_df(annotation_path=None):
    """유전자 주석 정보 로드"""
    if annotation_path is None:
        annotation_path = NUCLEOTIDE_ANNOTATION_PATH

    print(f"\n  [INFO] 유전자 주석 파일 로드: {annotation_path}")

    try:
        annotation_df = read_csv(annotation_path, sep=' ')
        print(f"  [OK] 로드 완료: {len(annotation_df)}개 유전자")

        # 유전자 목록 출력
        print(f"\n  [유전자 목록]")
        print(f"  {'Gene':<12} {'Start':>8} {'End':>8} {'Length':>8}")
        print(f"  {'-'*38}")
        for _, row in annotation_df.iterrows():
            length = row['end'] - row['start'] + 1
            print(f"  {row['gene']:<12} {row['start']:>8} {row['end']:>8} {length:>8}")

        return annotation_df
    except FileNotFoundError:
        print(f"  [ERROR] 파일을 찾을 수 없음: {annotation_path}")
        return None


def make_bedgraph(mutClustInput_df, cluster_df, output_file, annotation_path=None):
    """
    클러스터 결과를 시각화하여 bedgraph 이미지 생성

    Args:
        mutClustInput_df: 변이 입력 데이터
        cluster_df: 클러스터 결과 데이터프레임
        output_file: 출력 파일 경로
        annotation_path: 유전자 주석 파일 경로 (선택)
    """
    print(f"\n  {'='*60}")
    print(f"  시각화 생성")
    print(f"  {'='*60}")

    start_time = time.time()

    color_list = ['maroon', 'r', 'coral', 'chocolate', 'orange', 'gold', 'olive',
                  'yellow', 'lawngreen', 'palegreen', 'forestgreen', 'lime',
                  'mediumaquamarine', 'aquamarine', 'teal', 'aqua', 'steelblue',
                  'slategrey', 'cornflowerblue', 'blue', 'slateblue', 'indigo',
                  'plum', 'magenta', 'deeppink', 'pink']

    col_list = ['H-score']

    print(f"  [STEP 1] 데이터 준비")
    print(f"    • 입력 데이터: {len(mutClustInput_df)} positions")
    print(f"    • 클러스터 수: {len(cluster_df)}개")

    if annotation_path:
        gene_df = read_csv(annotation_path, sep=' ')
    else:
        gene_df = get_GeneInfo_df()

    if gene_df is None:
        print(f"  [ERROR] 유전자 주석을 로드할 수 없어 시각화를 중단합니다.")
        return

    input_df = mutClustInput_df[col_list]

    print(f"\n  [STEP 2] 그래프 생성")
    plt.figure(constrained_layout=True, figsize=(12, 8))
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(13.5, 3))
    axes = [ax] if len(col_list) == 1 else ax

    for ax, col in zip(axes, input_df):
        plot_df = input_df[col]
        plot_df.plot.bar(width=30, ax=ax, xticks=np.arange(0, 29903, 1000), ylabel=col)
        ax.tick_params(labelrotation=45, labelsize=5)

        # 클러스터 영역 표시
        cluster_count = 0
        for i, row in cluster_df.iterrows():
            ax.axvspan(row['left_position'], row['right_position'],
                      facecolor='lightgrey', alpha=0.8)
            cluster_count += 1

        print(f"    • 표시된 클러스터 영역: {cluster_count}개")

    print(f"\n  [STEP 3] 유전자 주석 추가")
    pre_y = 0
    gene_count = 0
    for i, row in gene_df.iterrows():
        trans = matplotlib.transforms.blended_transform_factory(
            axes[-1].transData, fig.transFigure
        )
        r = matplotlib.patches.Rectangle(
            (row['start'], 0.02),
            row['end'] - row['start'] + 1,
            0.02,
            facecolor=color_list[i],
            transform=trans,
            edgecolor='black',
            lw=0.5
        )
        fig.add_artist(r)
        text_y = 0.01
        if ((row['start'] - row['end']) < 1000) and (pre_y == 0.01):
            text_y = 0.05
        pre_y = text_y
        fig.text(
            (row['start'] + row['end']) / 2,
            text_y,
            row['gene'],
            ha='center',
            va='center',
            fontsize=5,
            fontweight='bold',
            zorder=10,
            transform=trans
        )
        gene_count += 1

    print(f"    • 표시된 유전자: {gene_count}개")

    print(f"\n  [STEP 4] 이미지 저장")
    plt.tight_layout()
    plt.savefig(output_file, format='png', dpi=150)
    plt.close()

    elapsed = time.time() - start_time

    print(f"  [OK] 시각화 저장 완료: {output_file}")
    print(f"  [TIME] 소요 시간: {elapsed:.2f}초")

    # 파일 크기 확인
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        print(f"  [INFO] 파일 크기: {file_size / 1024:.1f} KB")


def analyze_clusters_by_gene(cluster_df, annotation_path=None):
    """
    클러스터가 어떤 유전자에 위치하는지 분석

    Args:
        cluster_df: 클러스터 결과 데이터프레임
        annotation_path: 유전자 주석 파일 경로

    Returns:
        유전자별 클러스터 통계
    """
    print(f"\n  {'='*60}")
    print(f"  유전자별 클러스터 분석")
    print(f"  {'='*60}")

    if annotation_path:
        gene_df = read_csv(annotation_path, sep=' ')
    else:
        gene_df = get_GeneInfo_df()

    if gene_df is None:
        return None

    gene_clusters = {}

    for _, gene_row in gene_df.iterrows():
        gene_name = gene_row['gene']
        gene_start = gene_row['start']
        gene_end = gene_row['end']

        # 이 유전자와 겹치는 클러스터 찾기
        overlapping = []
        for _, cluster_row in cluster_df.iterrows():
            c_left = cluster_row['left_position']
            c_right = cluster_row['right_position']

            # 겹침 확인
            if not (c_right < gene_start or c_left > gene_end):
                overlapping.append({
                    'left': c_left,
                    'right': c_right,
                    'mutations': len(str(cluster_row['mut_positions']).split(','))
                })

        gene_clusters[gene_name] = {
            'start': gene_start,
            'end': gene_end,
            'length': gene_end - gene_start + 1,
            'cluster_count': len(overlapping),
            'clusters': overlapping
        }

    # 결과 출력
    print(f"\n  {'Gene':<12} {'Start':>8} {'End':>8} {'Length':>8} {'Clusters':>10}")
    print(f"  {'-'*50}")

    for gene_name, info in gene_clusters.items():
        print(f"  {gene_name:<12} {info['start']:>8} {info['end']:>8} "
              f"{info['length']:>8} {info['cluster_count']:>10}")

    # 클러스터가 가장 많은 유전자
    max_gene = max(gene_clusters.items(), key=lambda x: x[1]['cluster_count'])
    print(f"\n  [INFO] 최다 클러스터 유전자: {max_gene[0]} ({max_gene[1]['cluster_count']}개)")

    return gene_clusters


def print_summary_statistics(mutclust_input_df, ccm_count, cluster_count):
    """
    전체 분석 요약 통계 출력

    Args:
        mutclust_input_df: 입력 데이터프레임
        ccm_count: CCM 개수
        cluster_count: 최종 클러스터 개수
    """
    print(f"\n  {'='*60}")
    print(f"  전체 분석 요약")
    print(f"  {'='*60}")

    total_positions = len(mutclust_input_df)

    print(f"\n  [입력 데이터]")
    print(f"    • 총 positions: {total_positions}")
    print(f"    • H-score 범위: {mutclust_input_df[HSCORE].min():.6f} ~ {mutclust_input_df[HSCORE].max():.6f}")
    print(f"    • Frequency 합계: {mutclust_input_df[FREQ].sum():,}")

    print(f"\n  [분석 결과]")
    print(f"    • CCM 선정: {ccm_count}개 ({ccm_count/total_positions*100:.2f}%)")
    print(f"    • 최종 클러스터: {cluster_count}개")
    print(f"    • 압축률: {ccm_count} → {cluster_count} ({(1-cluster_count/ccm_count)*100:.1f}% 감소)")


def display_image(image_path):
    """
    이미지를 base64로 인코딩하여 출력 (프론트엔드에서 렌더링)

    Args:
        image_path: 이미지 파일 경로
    """
    if not os.path.exists(image_path):
        print(f"  [ERROR] 이미지 파일을 찾을 수 없음: {image_path}")
        return

    with open(image_path, 'rb') as f:
        img_data = base64.b64encode(f.read()).decode('utf-8')

    # 특수 마커로 감싸서 프론트엔드가 인식할 수 있도록 함
    print(f"<!--IMAGE:data:image/png;base64,{img_data}:IMAGE-->")


if __name__ == '__main__':
    print("utils.py - MutClust 유틸리티 모듈")
    print("=" * 50)
    print("\n사용 가능한 함수:")
    print("  • mutation_filtering(): 유효 범위 변이 필터링")
    print("  • get_GeneInfo_df(): 유전자 주석 로드")
    print("  • make_bedgraph(): 클러스터 시각화")
    print("  • analyze_clusters_by_gene(): 유전자별 분석")
    print("  • print_summary_statistics(): 요약 통계 출력")
    print("  • display_image(): 이미지 출력")
