# 문서를 표현하는 NLP 딥러닝 모델 논문 정리

2020년 6월 22일

## 차례
* 문서 내용 학습을 위한 딥러닝 모델 일반
* 문서의 임베딩 벡터를 학습하기 위한 딥러닝 모델
* 문서 요약 모델
* 문서 분류 모델
* 문서 딥러닝 시의 문제점들과 관련 연구
* 한글과 전문어 처리
* 긴 문서 처리
* 지도 학습 레이블링 이슈

## 1. 문서 내용 학습을 위한 딥러닝 모델 일반

### 자기 회귀 모델
딥러닝 신경망 기반의 언어 학습 모델에서는 자기 회귀 (auto-regression) 방식의 모델들이 사용.

(자기 회귀는 앞 시퀀스에서 생성된 토큰이 다음 시퀀스의 입력으로 사용되는 구조를 의미)

Feedback 구조를 가진 순환 신경망(Recurrent Neural Networks, RNN) 모델 계열이 많이 사용.


<img width="438" alt="image" src="https://github.com/yoonforh/deeplearning/assets/1460967/b3a03a68-e22a-46e3-b281-9a9431b16174">

[그림1] 순환 신경망(RNN)의 순환 구조

RNN은 기울기 소멸(gradient vanishing) 현상 등의 문제로 학습이 잘 진행되지 않아서 이 문제를 해결하고 좀더 긴 시퀀스에 걸쳐 정보를 전달할 수 있는 모델들이 등장.
즉, 정보가 다음 순환 때 사라지지 않도록 차폐를 결정하는 게이트를 통해 직접적인 선형 연결을 갖는 구조로 LSTM과 GRU 두 가지 모델이 많이 사용.

LSTM(Long Short-term Memory) : ‘Long Short-term Memory’, Sepp Hochreiter; Jürgen Schmidhuber, 1997
GRU (Gated Recurrent Unit) : ‘Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation’, Kyunghyun Cho et al. 2014





![image](https://github.com/yoonforh/deeplearning/assets/1460967/bcdfe9dc-270d-41f6-978e-9daf820e3819)


[그림2] LSTM 구조


LSTM과 GRU가 기본 RNN 모델에 비해서 정보를 더 효과적으로 여러 시퀀스에 걸쳐 전달할 수 있었지만 가중치를 공유하는 특성 등으로 인해 긴 시퀀스에 걸친 정보들을 효과적으로 학습하지 못하는 현상을 극복하지 못하였고 여러 시퀀스에 걸친 정보를 학습하기 위한 주의집중(attention) 모델이 제안되었다.

Attention 모델 : ‘NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE’, Dzmitry Bahdanau et al., 2015

주의집중 구조는 기계 번역에 사용되는 모델인 인코더-디코더 구조에서 처음 적용되었으며 각 시퀀스 위치별 중요도를 표현하는 가중치 벡터라고 생각할 수 있다.

![image](https://github.com/yoonforh/deeplearning/assets/1460967/41b451f3-135f-4b53-b685-6a2499d1e51a)


[그림3] Encoder-Decoder 모델에 적용된 Attention 구조

주의집중 구조가 기존의 RNN 계열 모델들의 학습 가능한 시퀀스 수를 늘려주었지만, RNN 계열 모델들은 순차적 특성 상 앞 시퀀스의 결과가 나올 때까지 다음 시퀀스를 계산할 수 없어 병렬 처리가 어려웠다. 이에 따라 커다란 언어 모델을 학습하는 데는 한계가 있었는데 RNN 순환 구조를 제거하고 주의집중 구조만으로 만든 Transformer 모델이 제안되었다.

Transformer 모델 : ‘Attention is All You Need’, Ashish Vaswani et al. (Google Brain), 2017

![image](https://github.com/yoonforh/deeplearning/assets/1460967/ef4dd4fc-4cd4-422a-aa84-9b79c8924ce3)

[그림4] Transformer 모델 구조


Transformer 모델은 최근 대부분의 언어 모델들이 기반으로 삼고 있는 모델이며, 자기 주의집중(Self-Attention), 다중 헤드 주의집중(Multi-head Attention), 위치 인코딩 (Positional Encoding) 등에서 기존 모델들과 크게 다른 점을 보인다.

![image](https://github.com/yoonforh/deeplearning/assets/1460967/d76e6b6b-6640-4fb0-87c5-621525efb60e)

[그림5] Transformer 모델의 자기 주의집중과 다중 헤드 주의집중

자기 주의집중의 계산 방식은 크기 조정된 내적 주의집중(scaled dot-product attention)이라고 하는데 문장의 각 요소들을 가중치를 행렬곱하여 query와 key로 만든 후 이들의 내적을 구한 다음 크기를 조정한 다음 다시 value에 해당하는 값과 행렬곱을 하는방식으로 자기 주의집중을 계산한다.
이러한 자기 주의집중이 하나가 아니라 여러 개를 사용하기 때문에 다중 헤드 구조이다.


$SelfAttention(Q,K,V) = softmax \left( {QK^\top \over \sqrt{d_k}} \right)$

[수식1] Transformer 모델의 self-attention
 
언어 학습 모델
Transformer 논문은 인코더 신경망과 디코더 신경망을 가진 전형적인 기계 번역 신경망 모델을 제안하고 있다.
즉, 인코더에는 원본이 되는 텍스트의 언어를 입력하고 디코더에는 번역 대상본이 되는 텍스트의 언어를 입력하여 트레이닝한 후 원본 언어를 입력하면 번역 대상본 언어로 번역하게 하는 모델이다. (참고로 구글에서는 기계번역 작업에 인코더는 Transformer 기반 구조를 사용하지만, 디코더는 추론 성능 이슈로 여전히 RNN을 변형한 구조를 사용한다고 한다. https://ai.googleblog.com/2020/06/recent-advances-in-google-translate.html)

Transformer가 제안한 자기 주의집중 모델은 RNN 계열 모델의 학습 차원 제약을 해결하여 수많은 형태의 언어 학습 모델을 파생하였다.

BERT 모델
BERT 모델 : ‘BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding’, Jacob Devlin et al., 2018

언어 학습 모델은 보통 방대한 양의 텍스트를 사용하여 범용적인 언어 표현 모델을 학습하는 사전 학습(pre-training) 과정과 이 범용으로 사전 학습된 모델을 작은 특정 태스크에 특화된 데이터 셋에 대하여 조정하는 미세 조정(fine-tuning) 과정 두 가지 과정에서 각각 목적인 다른 성격의 학습된 모델들을 만들 수 있다.
BERT는 자연어 처리의 사전 학습 기술로 제안되었으며 다양한 자연어 처리 태스크에서 최고 수준의 결과를 보여주어 큰 주목을 받고 있다.
사전 학습(pre-training) 모델이므로 BERT 모델은 언어 표현을 출력으로 만들어내는 모델이며 Transformer 모델의 Encoder-Decoder 구조 중 Encoder 구조만을 사용한다.
BERT 모델이 기존과 다른 사전 학습 방법으로 채택한 아이디어는 크게 다음과 같다.

Transformer 기반 아키텍처 : 기존에 많이 사용하던 RNN 계열 모델이 아닌 Transformer 아키텍처를 채택.
쌍방향(bi-directional) 트레이닝 : 기존의 방식은 제시된 앞쪽 단어들을 사용하여 다음 단어를 예측하는 방식을 사용하였다. 하지만 BERT는 문장 중에서 일부 단어를 무작위로 마스킹하여 마스킹된 단어를 맞추는 방식으로 트레이닝을 하는데(Masked Language Model, MLM) 이 방식은 마스킹된 단어를 추정하기 위하여 앞쪽 단어들 정보와 뒷쪽 단어들 정보를 모두 동시에 사용하게 된다는 장점이 있어서 좀더 깊이있는 언어 모델을 만들 수 있다. (동시에 모두 사용한다는 측면에서 쌍방향이 아니라 전방향 혹은 무방향이라는 의견이 있음)
다음 문장 예측 (Next Sentence Prediction, NSP) : 문장들 간의 관계를 이해할 수 있도록 BERT는 트레이닝 단계에서 다음 문장 예측 기법을 함께 사용한다.

![image](https://github.com/yoonforh/deeplearning/assets/1460967/83f9e694-02b5-4246-8f5c-0381479ace79)


[그림6] BERT의 입력 임베딩

위 그림에서 BERT의 입력 문장은 크게 토큰, 세그먼트, 위치 3가지 요소의 임베딩으로 변환되어 사용된다. 
입력은 두 개의 문장으로 구성되며, 토큰 임베딩은 각 문장의 요소들에 해당하는 임베딩을 포함한다. (논문에서 텍스트를 임베딩으로 표현하기 위해 WordPiece 임베딩 방법을 사용)

WordPiece 모델 : ‘Japanese and Korean Voice Search’, Mike Schuster and Kaisuke Nakajima, 2012

세그먼트 임베딩은 문장을 분리하는 요소이다. BERT는 NSP를 위해 두번째 문장을 연속된 다음 문장을 선택하거나 혹은 무작위 문장을 선택하고 레이블로 IsNext 혹은 NotNext 값을 부여한다.
마지막 위치 임베딩은 문장 내에서의 각 토큰의 위치를 나타내는 요소로 Transformer에서 사용한 위치 계산을 그대로 사용.


$PE\left(pos, 2i \right) = \sin \left( { pos \over { 10000 { 2i \over d_{model} } } } \right), \quad PE\left(pos, 2i + 1\right) = \cos \left( { pos \over { 10000 { 2i \over d_{model} } } } \right)$

[수식2] Transformer 모델의 위치 임베딩 계산

BERT로 학습된 언어 모델(Language Model, LM)을 사용하면서 미세 조정(fine-tuning)이 필요한 경우에는 보통 BERT 모델 다음에 신경망 계층을 추가하여 특정 목적에 맞도록 적용하는 방식을 사용한다.

언어 학습 모델은 BERT 외에도 Transformer 기반의 GPT-2, GPT-3, Transformer-XL, XLM 등의 모델이 소개되었다.

Transformer-XL : ‘Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context’, Zihang Dai, Zhilin Yang et al., 2019
XLM : ‘Cross-lingual Language Model Pretraining’, Guillaume Lample, Alexis Conneau, 2019
GPT-2 : ‘Language Models are Unsupervised Multitask Learners’, OpenAI, 2019
GPT-3 : ‘Language Models are Few-Shot Learners’, OpenAI, 2020


2. 문서의 임베딩 벡터를 학습하기 위한 딥러닝 모델

문서의 내용에 기반하여 문서의 특성을 나타내는 표현을 학습하기 위한 방법으로 참조할 수 있는 모델은 크게 문서의 내용을 요약하는 딥러닝 학습 모델과 문서를 유한 개의 클래스로 분류하는 딥러닝 학습 모델이 있다.
문서 요약 모델과 문서 분류 모델의 최근 연구 동향을 살펴보고 과제 활용 가능성을 정리해본다.

2.1 문서 요약 모델
문서의 내용을 표현하는 벡터를 구하는 방식을 사용하는 모델 중 하나로 문서 내용을 요약하는 딥러닝 모델이 있다.
문서 내용을 요약하는 접근법에는 크게 추출 방식과 추상 방식이 있다.

추출(Extractive) 방식 : 기준에 따라 좀더 관심이 있는 문서의 일부들을 추출하여 문서 요약을 만드는 방법
추상(Abstractive) 방식 : 원 문서에서 정보를 추출하여 추출한 정보에 기반하여 새로운 문장들을 재구성하여 문서 요약을 만드는 방법

보통 추출 방식은 Tf-Idf(term frequency–inverse document frequency)와 word2vec 알고리즘 등을 사용하여 문장의 중요성을 평가하고 중요한 문장들을 취합하는 방식으로 구현이 된다.

TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)
IDF(t) = log_e(Total number of documents / Number of documents with term t in it)
[수식3] TF, IDF 계산식

Word2vec : ‘Efficient Estimation of Word Representations in Vector Space’, Tomas Mikolov et al., 2013

추상 방식은 딥러닝 방식에서 많이 연구되고 있으며 전체 문서 내용에서 중요한 정보를 추출한 후 그 압축된 정보에 기반하여 요약 정보를 재구성하는 방식으로 구현이 된다. 이 방식이 좀더 사람의 두뇌가 요약을 하는 방식과 유사하다.

Transformer 언어 모델 기반으로 추출 방식과 추상 방식을 혼용한 문서 요약 연구
Transformer 언어 모델 기반으로 추출 방식과 추상 방식을 혼용한 문서 요약 연구 : ‘On Extractive and Abstractive Neural Document Summarization with Transformer Language Models’, Sandeep Subramanian et al., 2019
위 연구는 Transformer 언어 모델 기반으로 추출 방식과 추상 방식을 혼합 사용하여 문서의 요약을 생성하려는 시도이다.

![image](https://github.com/yoonforh/deeplearning/assets/1460967/1f2203c9-197f-4526-9515-9f608a63fc76)


[그림 7]
 추출 방식과 추상 방식을 혼용한 문서 요약 모델의 트레이닝 및 추론 순서

이 연구는 과학 논문을 요약하기 위해 주요 문장을 추출한 다음 단락들을 일정한 순서(1. 도입부, 2. 추출된 문장들, 3. 초록 및 논문의 나머지 부분)로 재정렬한 후 Transformer 언어 모델을 사용하여 이 재정렬된 추출 문장들을 학습하는 모델.
실제 요약을 생성해내는 것은 학습된 Transformer 모델이 생성하기 때문에 추상 방식이지만 학습을 위해 요약과 순서 조정을 먼저 한다는 게 특징.]요약을 생성하는 추론 시에는 Transformer 언어 모델에 1. 도입부와 2. 추출된 문장들만 전달한다.
초록이 있는 과학 논문이 아닌 뉴스나 특허 문서와 같은 경우에는 도입부를 전체 문서로 대체해서 진행한다.

(1) 추출 모델
주요 문장을 추출하기 위한 추출 모델은 기본적으로 계층적인 접근을 하여 기본적으로는 [그림 8]의 모델을 따르지만 단어 수준의 인코딩 시에도 문장 수준의 인코딩과 마찬가지로 bi-LSTM encoder를 사용한다. 디코더는 동일하게 LSTM decoder를 사용.

![image](https://github.com/yoonforh/deeplearning/assets/1460967/fc7f686b-f7b6-4dcb-bf1a-c087c5f996aa)

[그림 8] 추출 모델의 단어와 문장에 대한 계층적 처리 구조 (‘Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting’, Yen-Chun Chen et al., 2018)

이 추출 모델은 N 개의 문장으로 이루어진 문서를 추출하기 위해서 Sentence Encoder에서 N 개의 문장 임베딩 벡터를 만든 다음, 이를 입력으로 받는 bi-LSTM을 사용하여 N 개의 문서 표현 벡터를 만들어내는 구조이다.
N 개의 문서 표현 벡터는 dot-product 방식의 attention 메커니즘을 사용하여 컨텍스트 벡터를 만들게 되며 원래의 은닉층 벡터와 컨텍스트 벡터를 concat하는 개념으로 새로운 은닉층 벡터를 만들어 LSTM decoder에 넘겨주게 된다.
트레이닝 시의 손실 함수는 ground-truth로 정의된 요약 문장을 잘 선택하는지에 대한 cross-entropy 값을 사용하며, 실제 추론 시에는 beam-search 방식을 사용한다.

(2) 추상 요약을 위한 언어 모델
언어 모델은 Transformer 모델을 직접 사용하였고, auto-regressive 방식으로 트레이닝. 즉, 기학습된 언어 모델을 활용하지 않음.
문서 표현 모델 바로 다음에 ground-truth로 사용될 요약 텍스트를 붙여서 학습시킴으로써 문서 표현과 요약 텍스트의 결합 분포로 트레이닝되도록 했으면 추론(inference) 시에는 문서 표현을 주면 요약 텍스트를 생성하는 방식을 사용. (일반적인 sequence to sequence 모델과 부합)

문서 내 문장이 매우 많을 경우 Transformer 모델로 한번에 볼수 있는 토큰의 갯수를 벗어나기 때문에 과학 논문과 같은 경우는 도입부를 요약을 생성할 수 있는 충분한 정보량을 가진 Proxy로 간주. 이렇게 하면 [그림 7]에서 보이듯이 도입부 뒷쪽의 정보는 ground-truth로 사용될 요약 텍스트 뒷쪽에 붙여서 결합 분포로 트레이닝되는 데에만 사용. (즉, 도입부 뒷쪽 내용은 언어 모델에만 기여를 하고 실제 요약을 생성하는 데는 기여를 하지 않도록 함)

(3) 이 논문을 참고할 때 주의할 부분
해당 모델은 문서 표현 모델은 전체 문장 개수를 N이라 했을 때 주의집중 방식을 사용하여 요약 문장을 위한 M (<< N)개의 벡터로 추출 가능하다. 하지만 하나의 벡터로 표현되는 것은 아니므로 이 방식으로 문서 표현 벡터를 직접 구할 수는 없다. 추가적인 레이어가 필요하다.
그외, 트레이닝 상에서 ground-truth 데이터를 필요로 하는 지도 학습의 공통된 어려움, 문장이 매우 많은 큰 문서를 학습할 때 Transformer 모델로 처리하기 어려운 문제 등을 완전히 해결하지 못했다.

2.2 문서 분류 모델
문서의 요약을 생성하기 위해서 문서를 언어 모델로 표현해야 하는 문서 요약 모델과 달리, 문서가 어떤 분류에 속하는지를 학습하는 문서 분류 모델은 문서의 언어 모델 표현을 직접적으로 학습할 필요가 없는 장점이 있다.

BERT를 활용한 문서 분류 연구
BERT를 사용한 문서 분류 : ‘DocBERT: BERT for Document Classification’, Ashutosh Adhikari et al., 2019

위 연구는 BERT를 문서 분류에 적용하려는 초기 연구이다. BERT를 문서 분류에 활용할 때 주의할 사항은 다음과 같다.

컨텐츠 분류 관점에서는 문장 구조는 덜 중요하다. 
문서는 일반적인 BERT 입력에 비해 더 길 가능성이 높다.
문서에 적용되는 분류 레이블이 여러 개일 수 있다.

해당 연구는 이러한 문제를 해결하기보다는 문제 제기에 그치고 있으며, 이러한 단점을 감안한 상태에서 BERT를 사용한 문서 분류가 유의미한 결과를 보여주는지를 검증한다.

비지도 학습으로 사전 학습(pre-training)된 BERT 모델의 마지막 은닉 계층에 Fully-connected 계층을 연결하고 이를 softmax 분류기를 추가로 연결한 다음 미세 조정(fine-tuning) 개념으로 추가 학습을 시킨다.
이때 최종 손실 함수는 단일 레이블일 경우 binary cross-entropy, 다중 레이블일 경우는 cross-entropy를 사용한다.
 
이 외에도 이 연구에서는 BERT 모델 사용하여 추론할 때의 연산 비용을 줄이기 위해 BERT 모델을 상대적으로 작은 bi-LSTM으로 정제하는 기법을 적용하여 파라미터 수를 30배수로 줄인다.

신경망에서 지식 정보 정제 : ‘Distilling the Knowledge in a Neural Network’, Geoffrey Hinton et al., 2015

일반적으로 분류를 목적으로 하는 신경망의 경우 logit을 확률로 변환하는 softmax 출력 계층을 가지게 되는데 정제 시에는 온도를 뜻하는 T 값을 조정하여 클래스별 확률 분포간의 부드러움을 결정한다.
정제 학습 시에는 높은 T 값을 사용하나, 트레이닝이 완료된 후 예측 시에는 T 값을 1로 사용한다.

$q_i = {{exp(z_i / T )} \over { \sum_j exp(z_j / T ) }}$

[수식4] 신경망 정제 학습시 사용되는 클래스별 확률에 대한 softmax 수식. 온도 T값이 클수록 더 부드럽게 클래스간 연결 분포를 이룬다.

실제 BERT 기반 분류 학습 시에는 분류 과정의 손실 함수(cross-entropy 혹은 binary cross-entropy)와 정제 과정의 손실 함수를 결합하여 미세 조정 단계에서 end-to-end로 학습한다.

$L = L_{classification} + \lambda L_{distill}$

[수식5] BERT 문서 분류 모델의 손실 함수

3. 문서 딥러닝 시의 문제점들과 관련 연구

3.1 한글과 전문어 처리, 문서에 포함된 불릿 등 기호 처리, 문서별 문장 형태 상이
보통의 기업에서 사용하는 문서의 텍스트는 워드 프로세서의 경우라고 하더라도 일상적으로 사용하는 언어와는 다른 형태가 많다. 특별한 기호나 불릿 등에 의해 분류된 텍스트들이 다수 포함되며 이로 인해 언어 모델 적용 시 오류가 다수 발생한다. 또 대부분의 언어 모델들이 학습 데이터의 영향을 받아 영문에 비해 한글에 대한 이해가 떨어지는 이슈가 있다.
문서별 문장 형태가 간결하게 함축되거나 만연체로 늘어쓰거나 하는 형태에 따라서도 영향을 받을 수가 있는데 이러한 특성은 문장을 잘 구분하는 문장 분리기를 특별하게 구현할 필요가 있으며, 언어 모델의 스타일 학습을 통해서 완화하는 방법이 있다.

감정과 시제를 고려한 텍스트 생성 : ‘Toward Controlled Generation of Text’, Whiting Hu et al., 2018

![image](https://github.com/yoonforh/deeplearning/assets/1460967/7e42e9ff-1d76-4410-b88d-f145eba95141)

[그림9] 감정과 시제를 고려한 텍스트생성 모델

위 논문에서는 문장별로 감정 코드 (및 시제) c를 조건화하여 문장을 생성하고 적대적인 식별 모델을 통해 감정 코드 c를 식별하도록 하였다.
감정 코드는 이산적인 값을 가지므로 이를 역전파를 통해 학습할 수 있도록 하기 위해 온도 값을 사용한 softmax 함수를 사용하여 근사함으로써 미분 가능하게 하였다.
원래의 문장에서 컨텍스트를 추출하고 다시 컨텍스트로부터 문장을 생성해내는 Encoder와 Generator 부분은 Variational Auto-Encoder의 Encoder, Decoder 역할에 기반하고 있으며, 각각 1계층 LSTM을 사용한다.
원 문장과 생성된 문장에서 각각 코드를 추정하는 Discriminator 네트웍은 Convolutional Network을 사용하며 트레이닝 시에는 Discriminator 네트웍과 Encoder/Generator 네트웍 부분을 번갈아서 트레이닝하는 방법을 사용한다.

한글 어절을 고려한 문장 분류 : ‘Integrated Eojeol Embedding for Erroneous Sentence Classification in Korean Chatbots’, 최동현 외 (카카오), 2020

![image](https://github.com/yoonforh/deeplearning/assets/1460967/ef319541-30f5-4370-8d98-36584e1da081)

[그림10] 통합 어절 임베딩 네트웍

카카오에서 발표한 논문은 한글이 포함된 문장에 오타가 있거나 잘못된 뛰어쓰기가 된 경우 심각하게 정확도가 떨어지는 문제를 해결하기 위해 어절을 부분 단어(subword)들로 나눠 먼저 각각의 임베딩 벡터들을 계산한 다음 이들을 하나의 어절 통합 임베딩 벡터를 만들기 위해 통합 어절 임베딩 네트웍을 통과시킨 다음 그 결과물인 임베딩 벡터를 토큰으로 사용하는 방안을 제안하고 있다.
각 어절은 4가지 유형으로 subword 리스트로 변환되는데 논문에서 사용한 유형들은 각각 자모로 분해, 문자 단위로 분해, BPE 방식으로 분해, 형태소로 분해 네 가지이다.
이 네 가지 종류의 subword 리스트가 [그림 10]의 네트웍으로 입력되어 결과적으로 어절을 표현하는 하나의 통합 임베딩 벡터로 변환되는 셈이다.

3.2 긴 문서 처리
긴 문서의 경우 문장들 개수가 많아서 self-attention으로 구현된 transformer 모델의 최대 시퀀스 수를 넘어가는 경우가 생긴다. 그리고 연산 시간, 메모리 사용량 등도 관련하여 발생하는 문제이다.

Transformer 모델로 긴 문서 처리하기 : ‘Longformer: The Long-Document Transformer’, Iz Beltagy et al, 2020 
GO ON with the paper

![image](https://github.com/yoonforh/deeplearning/assets/1460967/11e1332c-2583-4f6d-b17a-b806de9ce4f5)


[그림11] Longformer 모델의 sliding window attention 패턴들. (b),(c),(d) 세 가지 중 하나를 사용한다.

Longformer 모델은 Transformer 모델의 self-attention 연산 오버헤드를 줄이기 위해 sliding window attention을 도입한 모델이다. 논문에서는 총 세 가지 모델로 테스트가 되어 있다.
기본적인 sliding window attention이 있으며, receptive field를 넓히기 위해 Convolutional network처럼 dilation을 도입한 dilated sliding window attention 그리고, BERT와 같이 Masked Language Model을 지원하기 위해 [CLS], [SEP] 같은 특수 토큰의 경우에는 full attention을 사용하는 global+sliding window attention 혼합 패턴이 있다.
RoBERTa(‘RoBERTa: A Robustly Optimized BERT Pretraining Approach’, Yinhan Liu et al., 2019)와 같이 시퀀스 크기를 분할하여 제약하는 경우와 달리 BERT의 MLM을 그대로 지원할 수 있다는 장점이 있다.

![image](https://github.com/yoonforh/deeplearning/assets/1460967/a9a63d5c-d7c5-4537-bafb-86ef95aadb6e)

[그림12] Longformer 모델과 Transformer (Full self-attention) 모델의 연산 시간과 메모리 사용량 비교

Longformer 논문 저자들은 sliding window attention을 효과적으로 GPU로 계산하기 위해 CUDA 구현체를 제공한다. [그림12]의 Longformer 부분은 이 CUDA 구현체를 사용한 경우의 성능이다.

3.3 지도학습 레이블링 이슈
문서 요약 학습과 같은 경우 긴 문서에 대한 모범 답안이라고 할 수 있을 문서 요약본을 별도로 만들어야 지도 학습이 가능하다. 논문의 경우에는 초록이 별도로 있기 때문에 초록을 요약의 답안으로 제시할 수 있지만 대부분의 문서는 그렇지 못하다. 따라서 문서 요약본을 만드는 레이블링 작업이 학습의 선결 요건이 되는 문제가 발생하고, 레이블링을 하더라도 요약 작업은 사람마다 수준과 스타일이 달라서 그 품질 수준에 따라 트레이닝이 제대로 되는지가 좌우될 수 있다.

문장 임베딩을 사용하여 비지도 방식 텍스트 요약 : ‘Unsupervised Text Summarization Using Sentence Embeddings’, Aishwarya Padmakumar et al, 2018

위 논문에서는 문서의 요약을 비지도 방식인 클러스터링을 활용하여 요약 문장들을 생성할 것을 제안한다.
문장 임베딩은 (1) skip-gram 메소드가 단어 임베딩을 생성하기 위해 앞 단어와 뒷 단어의 관계를 학습했던 것처럼 앞 문장과 뒷 문장을 통해 현재 문장을 추정하는 학습 방식을 통해 문장 임베딩을 생성하는 Skip Thought Vectors 방식과 (2) 텍스트의 유사성이나 문장 연결 관계, 감정 분류 등의 지도 학습 모델을 통해 학습된 임베딩을 사용하는 paragram 방식 두 가지를 사용한다.

![image](https://github.com/yoonforh/deeplearning/assets/1460967/6178cc7f-2a59-49b1-8821-312fa8599067)

[그림13] 문장 임베딩을 생성하는 방법 중 하나인 Skip Thoughts 모델 (‘Skip-Thought Vectors’, Ryan Kiros et al, 2015)

논문에서는 이렇게 생성된 문장 임베딩을 K-Means와 같은 클러스터링 방법을 사용하여 적당한 클러스터로 분류한 다음, (1) 각 클러스터의 중심점에 가장 가까운 거리에 있는 문장 임베딩을 선택하여 문장을 요약하는 추출 방식과 (2) LSTM으로 구현된 디코더 네트웍을 사용하여 각 임베딩을 문장으로 변환해내는 추상 방식을 사용한다. (이 경우에서는 추상 방식이 특별히 다른 임베딩 벡터를 통해 문장을 생성하는 것이 아니므로 추출 방식보다 정확도만 떨어지는 결과를 가져오는 것으로 의심된다.)
벡터 공간 상에서 클러스터를 이루는 문장들이 서로 가까운 의미를 가질 것이므로 각 클러스터에서 대표 문장만 가져오면 요약을 구성하는 데 충분할 것이라는 아이디어이다.

3.4 시스템 사양 이슈
대부분의 언어 모델의 기반이 되는 Transformer 모델은 self-attention 방식을 택하고 있는데 self-attention은 최대 시퀀스 수 n의 제곱에 수렴하는 시간 및 공간 복잡도를 가지고 있다.
따라서 시퀀스가 늘어나면 훨씬 더 많은 계산량과 메모리량을 요구하게 되는 문제가 있다.
문서를 학습하기 위해서는 최대 문장 개수만큼의 시퀀스를 처리해야 할 수도 있는데 문서에 포함된 문장 개수는 매우 가변적이며 큰 문서 즉, 문장이 매우 많은 문서도 존재한다.

선형 복잡도 수준으로 Self-Attention 구현하기 : ‘Linformer: Self-Attention with Linear Complexity’, Sinong Wang et al., 2020

Facebook AI에서 2020년 6월에 발표한 Linformer 모델은 Transformer 모델의  복잡도를  즉, 선형 수준의 복잡도로 변환해주는 수정 모델에 대한 연구이다.

$head_i = Attention(QW_i^Q,KW_i^K,VW_i^V) = softmax \left[ {QW_i^Q(KW_i^K)^\top \over \sqrt{d_k}} \right] VW_i^V$

[수식6] Transformer 모델의 self-attention 수식

위 수식에서 softmax 부분을 계산한 결과값은 최대 시퀀스 길이를 이라고 하면  차원의 행렬 가 된다. 
행렬  계산은 두 개의  행렬을 행렬곱하는 연산을 포함하므로  복잡도를 가진다.
Linformer 모델에서는  차원의 선형 프로젝션을 사용하여 행렬 를  차원을 가지는 행렬 로 대체할 수 있다는 주장을 한다.

$\overline{head_i} = Attention(QW_i^Q,E_iKW_i^K,F_iVW_i^V) = softmax \left[ {QW_i^Q(E_iKW_i^K)^\top \over \sqrt{d_k}} \right] \cdot F_iVW_i^V$

[수식7] Linformer 모델의 self-attention 수식

Linformer 모델의 self-attention 수식에 포함된 , 는 각각  차원을 가지는 선형 프로젝션 행렬이다.

다음 [그림 14]에서는 Linformer 모델의 선형 프로젝션 구조를 표현하고 있으며 Transformer 모델과 동일한 시퀀스 길이와 배치 크기를 사용했을 때의 추론 시간을 비교한다.

![image](https://github.com/yoonforh/deeplearning/assets/1460967/5b40d8bf-98e6-4202-84ff-5e91ad0e9bb5)

[그림14] Linformer 모델의 선형 프로젝션을 사용한 self-attention 구조 및 Transformer 모델과의 추론 시간 비교

이외에도 Transformer 모델의 추론 시간을 줄이기 위해 혼합 부동 소수점 정확도 사용, 더 작은 모델로 증류(distillation)하기 등을 혼합해서 사용해서 클라이언트에서도 탑재하여 추론할 수 있는 모델을 제안하고 있다.

