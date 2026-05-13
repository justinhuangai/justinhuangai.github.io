---
title: "《Attention Residuals》：讓殘差連接也注意力化"
date: "2026-03-19T16:49:27+08:00"
category: "Technical Report Reading"
description: "Kimi Team 的 Attention Residuals 技術報告：為什麼殘差連接也該「注意力化」，以及 Full AttnRes / Block AttnRes 如何把這個想法做成可訓練、可部署的系統"
tags: [technical-report-reading, residual-connections, transformer, AI, LLM, python]
pinned: false
---

2026 年 3 月 16 日，Kimi Team 在 arXiv 上傳了一篇技術報告：[《Attention Residuals》](/papers/2603.15031v1.pdf)（注意力殘差）。

從整份報告的結構就能看出作者真正的重心：不是只發一個新模組，而是按 `motivation -> AttnRes -> Block AttnRes -> infrastructure -> experiments -> discussion` 的順序，把「殘差連接到底在做什麼」這件事重新講了一遍。

## 0. 先認幾個詞

如果你完全沒有機器學習背景，可以順著這篇報告真正關心的問題，按下面這個順序先建立一個直覺：

- `Transformer`：今天大多數大模型的基礎架構。你可以先把它理解成一台一層一層處理資訊的機器。
- `hidden state`：模型在某一層裡的內部中間表示。可以粗略理解成「模型此刻腦子裡的臨時筆記」。
- `residual connection / 殘差連接`：層與層之間的一條「保留舊內容」的通道。它會先把上一層的內容留住，再把這一層新算出來的東西加上去。
- `residual / 殘差`：更接近「這一層新補上去的增量」，也就是上面那條殘差連接裡新增的那一部分。
- `attention`：從很多資訊裡，挑出「當前最該看哪一部分」的機制。你可以先把它記成「有選擇地看重點」。
- `PreNorm`：在進入一層之前，先把數值尺度調勻，再做後續計算。可以把它想成「先把音量調到合適，再繼續混音」。

## 1. 一句話說清楚

這份技術報告提出了一個問題：

**既然 Transformer 已經用 attention 取代了「序列維度上的遞迴」，為什麼大模型在「深度維度上的資訊聚合」還停留在固定加法？**

現代 LLM 幾乎都在用一種很常見的層結構：先做 PreNorm，再走 residual。直白地說，就是先把數值尺度調勻，再把這一層新算出來的結果加回原輸入。大家熟悉它的一個功能，是讓訓練過程更穩定，深層網路不那麼容易失控。但作者提醒我們，殘差連接其實還有另一個同樣重要、卻長期被忽視的角色：

**它定義了資訊怎樣沿著深度被彙總。**

如果下面的式子看不熟，不用卡住，直接看後面的「翻譯成人話」就夠了。

標準殘差的規則很簡單：

$$
h_l = h_{l-1} + f_{l-1}(h_{l-1})
$$

這裡可以直接把兩部分拆開看：

- $h_{l-1}$：舊內容，也就是上一層已經有的表示
- $f_{l-1}(h_{l-1})$：這一層新算出來的增量，更接近「殘差」這個詞本身

而把這兩部分重新加在一起的整條做法，才更準確地叫「殘差連接」。

把這個遞推式展開，你會得到：

$$
h_l = h_1 + \sum_{i=1}^{l-1} f_i(h_i)
$$

翻譯成人話就是：第 $l$ 層看到的輸入，本質上是「embedding 加上前面所有層輸出的統一加總」。每一層的權重都是 1，沒有選擇，沒有抑制，沒有「這一步我更該看第 3 層還是第 17 層」的機制。

AttnRes 的核心思想只有一句話：

**把 residual 從固定加法，改成沿深度做一次 softmax attention。**

## 2. 舊殘差到底哪裡有問題？

這份技術報告最重要的地方，不在於它提出了一個新公式，而在於它把一個大家已經習慣了的東西重新問題化了。

標準殘差長期被視為「訓練穩定性工具」。只要能讓梯度過得去，它就算完成任務了。但從資訊流角度看，這條路徑其實非常粗糙。

想像你在寫一份持續迭代的文件。每一輪修改，你都不是「挑出最 relevant 的舊版本內容再整合」，而是把之前所有版本一股腦全文追加到文件末尾。第 20 輪的時候，前 3 輪的重要洞察當然還在，但它們已經淹沒在越來越厚的堆疊裡了。

PreNorm 的問題就在這裡。報告引用了 SiameseNorm 的觀察，並進一步強調：在 PreNorm 下，`hidden state` 的量級會隨著深度近似按 $O(L)$ 增長。這裡的 hidden state，說白了就是模型每一層裡那份「內部筆記」。結果就是：

- 越往後的層，看到的是一個越來越膨脹的「歷史總和」
- 早期層的資訊雖然沒有消失，但會被不斷稀釋
- 後面層如果還想「發出聲音」，就被迫輸出更大的量級

這篇技術報告把這個現象叫 `PreNorm dilution`。這是一個非常準確的命名。不是梯度斷了，不是模型炸了，而是每一層的相對貢獻被越來越稀。

報告裡有一條值得抓住的潛台詞：我們在序列維度上早就不滿足於「所有過去 token 一視同仁」了，所以才有了 attention；那為什麼到了深度維度，卻還能接受「所有過去層統一權重相加」？

## 3. AttnRes 到底做了什麼

AttnRes 的形式很乾淨。第 $l$ 層不再機械地接收「前面所有層輸出的總和」，而是對這些歷史表示做一次加權選擇：

$$
h_l = \sum_{i=0}^{l-1} \alpha_{i \to l} \cdot v_i
$$

其中權重 $\alpha_{i \to l}$ 來自一層 softmax。你可以先把 softmax 理解成「把一組分數壓成一組權重，而且所有權重加起來等於 1」，這樣模型才能明確表達「更該看誰、少看誰」：

$$
\alpha_{i \to l} = \operatorname{softmax}\left(w_l^T \operatorname{RMSNorm}(k_i)\right)
$$

如果你沒接觸過 attention，還有一個最省力的理解方式：

- `query`：當前這一層現在想找什麼
- `key`：每一層歷史資訊各自貼著什麼「索引標籤」
- `value`：最後真正被取回來、參與彙總的內容

這裡最值得注意的設計有三個。

第一，**query 不是當前 hidden state 現算出來的，而是每層一個可學習的 pseudo-query 向量 $w_l$。**  
這有點反直覺。我們平時看到 attention，會自然以為 query 必須來自當前輸入。但作者故意把 query 設計成層級參數，而不是 token 級動態向量。這樣做的好處是：同一個 block 裡的多個 query 可以提前批量算，後面的基礎設施優化才有空間做。

第二，**key/value 直接來自前面層的輸出。**  
也就是說，真正帶來「輸入相關性」的不是 query，而是各層當前樣本上的表示本身。不同樣本經過前面層後得到的 key 不一樣，所以最後的深度注意力依然是 input-dependent 的。

第三，**key 前面加了 RMSNorm。**  
這是個很關鍵的小設計。因為如果不做歸一化，量級大的層會天然在點積裡佔便宜，你得到的就不是「誰更 relevant」，而更像「誰聲音更大」。

用 Python（基於 PyTorch）寫出來，大概就是這樣：

```python showLanguage
import torch
from torch import nn


def attention_residual(
    sources: list[torch.Tensor],
    pseudo_query: torch.Tensor,
    norm: nn.RMSNorm,
) -> torch.Tensor:
    keys = torch.stack([norm(source) for source in sources], dim=0)
    values = torch.stack(sources, dim=0)

    logits = keys @ pseudo_query
    weights = torch.softmax(logits, dim=0)
    return (weights.unsqueeze(-1) * values).sum(dim=0)
```

這個式子看上去像是「把 attention 用在 residual 上」。但更準確的說法是：

**它把 residual connection 從「固定的累加器」改成了「可選擇的深度檢索器」。**

## 4. 這份報告給了想法，也給了工程

**一句話結論：這篇報告提出了 Full AttnRes，它把這個想法推進成了一套可訓練、可部署、算得清帳的工程方案。**

Full AttnRes 讓每一層都 attend 到前面所有層，理論上很好理解，實際上也不算太貴。因為網路深度 $L$ 通常遠小於序列長度 $T$，所以作者說，單純算術量 $O(L^2 d)$ 並不是最可怕的問題。

真正的問題出現在大訓練裡：

- activation recomputation 會把本來可以丟掉的中間層輸出重新變成必須保存的對象
- pipeline parallelism 會讓這些跨層表示需要跨 stage 傳輸
- 一旦每層都要看所有前層，通訊和快取壓力會快速上去

所以他們又提出了 **Block AttnRes**。

做法是把 $L$ 層切成 $N$ 個 block。block 內部先用普通求和攢成一個 block representation，跨 block 再做 attention。這樣一來：

- Full AttnRes：看的是所有歷史層
- Block AttnRes：看的是所有歷史 block 的摘要，再加當前 block 的部分和

本質上是用「摘要級跨層注意力」換取可擴展性。

這還沒完。作者不是只說「我們分塊了，所以省記憶體」，而是真的把系統層的帳也算清楚了：

- 訓練階段用 **cross-stage caching**，避免 pipeline 裡重複傳歷史 block
- 推理階段用 **two-phase computation**
- 第一階段並行算 inter-block attention
- 第二階段順序算 intra-block lookback，再用 online softmax merge 合併

從附錄和 `table/memory_access.tex` 裡能看到最硬核的一組數字。按報告給的典型設定：

- 標準 residual：每層 residual 機制 I/O 是 `3d`
- naive Full AttnRes：`130d`
- 優化後的 Full AttnRes：`24d`
- Block AttnRes：`5.5d`
- mHC：`34d`

這組數字特別說明問題。Block AttnRes 不是「便宜到跟標準 residual 一樣」，但它已經從「明顯不現實」降到了「工程上值得試」。而且報告實測給出的代價也不大：

- 訓練端 wall-clock overhead 小於 4%
- 推理端 latency overhead 小於 2%

這也是它像一篇系統級技術報告的原因。很多 paper 的問題在於「idea 是新的，帳是糊的」；這篇在帳本上反而做得很用力。

## 5. 實驗最該看什麼

**比起主實驗裡個別分數的升降，更重要的是 AttnRes 在縮放趨勢、訓練動力學和下游能力上都給出了方向一致的信號。**

### 5.1 縮放定律：不是偶然贏一把

作者先做了五個模型規模的 scaling law 實驗，對比 Baseline、Full AttnRes 和 Block AttnRes。

擬合出來的曲線是：

- Baseline：$1.891 \times C^{-0.057}$
- Block AttnRes：$1.870 \times C^{-0.058}$
- Full AttnRes：$1.865 \times C^{-0.057}$

這三條曲線最重要的資訊不是「斜率差了多少」，而是：

**AttnRes 在整個 compute 範圍裡都持續更低。**

報告給了一個很容易傳播的結論：在 `5.6 PFLOP/s-days` 這個預算點，Block AttnRes 的 loss 相當於 baseline 多花 `1.25x` 算力才能達到的水準。

換句話說，這不是「在某個模型大小上碰巧調對了」，而是有比較穩定的規模收益。

### 5.2 大模型主實驗：不是玩具規模

主實驗不是小模型 toy benchmark，而是基於 Kimi Linear 的一個大配置：

- `48B total / 3B activated parameters`
- 27 個 Transformer blocks，也就是 54 層
- 8-of-256 routed experts + 1 個 shared expert
- 預訓練 `1.4T tokens`

這說明作者不是只在「小模型上做漂亮曲線」，而是真的把這個 residual 改造塞進了一個大訓練配方裡。

### 5.3 訓練動態中，輸出量級不再失控

Baseline 的 output magnitude 會隨著深度一路漲上去。訓練動態圖裡給的數值非常誇張：從前面 block 的 `0.04`、`0.06`、`0.10`，一路漲到後面 block 的 `10.47`、`12.15`。這就是 PreNorm dilution 的視覺化版本。

Block AttnRes 則完全不是這條曲線。它在 block 邊界形成一種週期性重置，量級大致在 `0.21` 到 `1.91` 之間波動，沒有出現一路失控上揚。

這非常重要，因為它說明 AttnRes 不是只在最後 benchmark 上「多拿了幾分」，而是真正在訓練動力學層面改變了表示如何沿深度堆積。

### 5.4 下游任務：提升最明顯的是推理和程式碼

預訓練後，AttnRes 在報告列出的全部評測上都不差於 baseline，幾個最亮眼的點包括：

- MMLU：`73.5 -> 74.6`
- GPQA-Diamond：`36.9 -> 44.4`
- Math：`53.5 -> 57.1`
- HumanEval：`59.1 -> 62.2`
- C-Eval：`79.6 -> 82.5`

這裡最值得注意的是 GPQA、Math、HumanEval 這種多步推理或程式生成任務漲幅更大。報告作者的解釋是：如果後層能更有選擇地回收前層表示，那麼 compositional tasks 會更受益。這個解釋是說得通的。

因為複雜推理最怕的不是「資訊不存在」，而是「資訊在網路很深的地方被埋住了」。

## 6. 消融實驗告訴了我們什麼

**消融實驗的關鍵結論，不是「連得更密就更強」，而是「沿深度做輸入相關的選擇性聚合」這件事本身在起作用。**

這份報告的消融做得不錯，因為它不只是證明「有用」，還試圖證明「為什麼有用」。

幾個最有意思的結論：

- **DenseFormer 1.767，幾乎和 baseline 1.766 一樣。**  
  這說明「能訪問所有前層」本身還不夠，關鍵在於權重是不是 input-dependent。

- **mHC 到了 1.747，已經明顯變好。**  
  這說明深度維度上的動態混合確實有效。

- **Full AttnRes 到了 1.737。**  
  它比 baseline、DenseFormer、mHC 都更低，說明顯式的 softmax depth attention 是更強的一條路。

- **SWA（只看最近窗口）只有 1.764。**  
  這很有價值。它說明 AttnRes 的收益不只是「多看最近幾層」，而是「能選擇性地看更遠的層」。

- **Block size 從 2、4、8 變化時，loss 都在 1.746 左右。**  
  這就是為什麼作者最後固定大約 8 個 blocks。不是拍腦袋，而是工程和效果之間一個相當好的 sweet spot。

- **input-dependent query 版本做到 1.731，比 Full AttnRes 還好。**  
  這一點非常耐人尋味。它說明當前報告裡的 pseudo-query 設計並不是性能上限，而是一個為基礎設施優化讓路的折衷。也就是說，作者不是不知道更強的寫法，而是主動選了更容易擴展的寫法。

這正是這份報告有意思的地方。你從正文、消融和系統設計裡能更清楚地看到他們的真實取捨：不是盲目追求最低 loss，而是在追求「足夠強，同時真能訓起來」。

## 7. 怎麼看這份報告

第一，這份報告最重要的，不是它發明了一個新模組，而是它把 residual connection 從「訓練穩定性工具」重新提升成了「資訊路由機制」。

這個視角一旦建立起來，很多東西都會被重新理解。殘差不再只是 gradient highway，它還是 depth aggregation rule。你會開始追問：

- 每一層到底能不能選擇性地訪問前層？
- 深度維度上有沒有 attention sink？
- 舊的 residual 變體本質上是不是 depth-wise linear attention？

而這正是報告討論部分厲害的地方。作者把一堆殘差變體統一進了一個 `depth mixing matrix` 的視角裡，進一步指出：

**很多已有方法，本質上都像是在深度維度上做 linear attention；AttnRes 做的是 depth-wise softmax attention。**

這個說法很大膽，但也很有啟發性。它等於是在說：Transformer 當年把序列維度從 recurrence 推到了 softmax attention；AttnRes 試圖把深度維度也推進一步。

第二，這篇技術報告的氣質很像「先把問題提對，再把系統做順」。它沒有執著於把每個部件都做到最 fancy。比如 query 故意做成 layer-specific 參數，而不是 token-dependent 動態向量，性能上未必絕對最強，但它給了 batching、two-phase computation、pipeline cache 一個成立的基礎。很多時候，一篇能落地的技術報告，靠的不是最激進的局部設計，而是整體約束下的取捨。

第三，這份報告中的這句話：

**Why is depth-wise aggregation still fixed while everything else has become adaptive?**

說得挺好。

## 8. 這份報告的邊界

第一，它目前是 **technical report / arXiv preprint**，不是已經過同行評審的會議論文。寫這類文章時，最穩妥的態度不是「它已經證明了未來」，而是「它提出了一個很強的視角，並給出了一套有工程可行性的實作」。

第二，它的大規模結果主要建立在 Kimi Linear 這條架構線上：MoE、KDA/MLA 混合注意力、Moonlight / DeepSeek-V3 風格訓練配方。雖然這不削弱結果本身，但也意味著我們還不能自動把結論外推到所有 dense decoder-only Transformer。

第三，報告自己也承認：Full AttnRes 其實更強，Block AttnRes 是今天硬體約束下的工程解。未來如果顯存、頻寬、interconnect 再往前走，或者更高效的 depth attention 變體出現，今天這版 Block 設計很可能不是終點。

## 9. 最後的感受

如果把過去十年大模型架構的主線粗暴地概括一下：

- Seq2Seq 在問：怎麼把一個序列壓成另一個序列？
- Bahdanau 在問：為什麼生成時不能回頭看輸入的不同位置？
- Transformer 在問：為什麼序列建模一定要靠遞迴？
- Chinchilla 在問：為什麼更多算力一定主要砸到參數量上？

那《Attention Residuals》（注意力殘差）問的是：

**為什麼深度上的資訊彙總，還停留在「所有歷史層統一加總」的時代？**

這個問題問出來，本身就已經很有價值。

幾年後 AttnRes 會不會像 PreNorm 一樣成為預設配置還不確定，但這篇技術報告已經把 residual connection 重新變成了一個值得被思考、被設計、被優化的對象。

以前大家說 attention 改寫了序列建模。

這份技術報告在嘗試改寫 residual。

2026 年春，Kimi 團隊的工作已經說明：當Scaling Laws 開始顯出逼近瓶頸的跡象時，LLM 的結構創新將持續湧現。

---

**延伸閱讀**

- [《Sequence to Sequence Learning with Neural Networks》](/zh-TW/posts/sequence-to-sequence-learning-with-neural-networks/)（使用神經網路進行序列到序列學習） — 編碼器-解碼器範式的確立
- [《Neural Machine Translation by Jointly Learning to Align and Translate》](/zh-TW/posts/neural-machine-translation-by-jointly-learning-to-align-and-translate/)（通過聯合學習對齊與翻譯實現神經機器翻譯） — 注意力機制的起源
- [《Attention Is All You Need》](/zh-TW/posts/attention-is-all-you-need/)（注意力就是你所需要的全部） — 注意力成為主角，Transformer 的誕生
- [《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》](/zh-TW/posts/bert/)（BERT：用於語言理解的深度雙向 Transformer 預訓練） — 預訓練範式的確立
- [《Scaling Laws for Neural Language Models》](/zh-TW/posts/scaling-laws-for-neural-language-models/)（神經語言模型的縮放定律） — 規模的數學
- [《Language Models are Few-Shot Learners》](/zh-TW/posts/language-models-are-few-shot-learners/)（語言模型是少樣本學習者） — 更大的模型，更善於從上下文中誘發能力
- [《Training Compute-Optimal Large Language Models》](/zh-TW/posts/training-compute-optimal-large-language-models/)（訓練算力最優的大型語言模型） — 怎樣花算力最划算
