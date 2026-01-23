## **公式 F1：能量密度收敛（自旋链热力学极限）**

**LCL 形式：**

\[
\Phi_1 := \ell_{\text{highE}} \ \wedge\ E\,G\,\ell_{\text{lowE}}
\]

**物理背景与意义：**

用于检验某类自旋链（例如 AKLT / gapped 自旋链）随着链长 N 增大，**能量密度**从短链的边界效应逐渐进入 **bulk 稳定区间**：即“起初能量密度偏高，但最终会进入低能量密度并一直保持”。

**标签（谓词）建议：**

- \(\ell_{\text{highE}}\)：当前长度 N 的能量密度 \(e(N) > e_\star + \delta\)
- \(\ell_{\text{lowE}}\)：当前长度 N 的能量密度 \(e(N) < e_\star + \delta\)
  其中 \(e_\star\) 是你定义的目标 bulk 能量密度（可用大 N 数值近似），\(\delta\) 是容差。

**验证流程：**

1. 对每个 N（例如从 N=4 到 N=\(N_{\max}\)），用 MPS 算期望能量密度 \(e(N)\)。
2. 给每个 N 打标签 \(\ell_{\text{highE}}, \ell_{\text{lowE}}\)。
3. 用模型检测验证 \(\Phi_1\)：应当能找到某个 \(N_0\)，使得所有 \(N \ge N_0\) 都满足 \(\ell_{\text{lowE}}\)。

---

## **公式 F2：能隙不坍塌（相稳定性 / 无相变）**

**LCL 形式：**

\[
\Phi_2 := G\big(\neg \ell_{\text{gap}\downarrow} \ \vee\ X \neg \ell_{\text{gap}\downarrow}\big)
\]

等价于“全局上：如果当前不是小能隙（不是近似 gapless），那么下一步长度也不是小能隙”。

**物理背景与意义：**

检验 gapped 相（如 AKLT、cluster 等）在增大 N 的过程中，不出现“突然闭隙”的异常现象；即相位不发生相变。

**标签建议：**

- \(\ell_{\text{gap}\downarrow}\)：能隙 \(\Delta(N) < \Delta_{\min}\)

**验证流程：**

1. 对每个 N 估计 \(\Delta(N)\)（可以基于 parent Hamiltonian 的小规模 exact / DMRG / MPS 变分，或你论文里允许的近似方法）。
2. 给每个 N 打 \(\ell_{\text{gap}\downarrow}\)。
3. 用模型检测验证 \(\Phi_2\)。若失败，反例会给出具体某步 \(N \to N+1\) 闭隙的“断点”。

---

## **公式 F3：关联长度/关联衰减（clustering）**

**LCL 形式：**

\[
\Phi_3 := E\,G\,\ell_{\text{corr}\downarrow(d)}
\]

**物理背景与意义：**

检验 gapped 自旋链的**指数关联衰减**：当链足够长后，固定距离 d 的两点关联会始终很小，并且随着 N 增大不会反弹。

**标签建议：**

- \(\ell_{\text{corr}\downarrow(d)}\)：对于某固定距离 d，满足
  \[
  \big|\langle O_i O_{i+d}\rangle - \langle O_i\rangle \langle O_{i+d}\rangle\big| < \varepsilon
  \]
  其中 \(O\) 可选 \(Z\)、或自旋算符 \(S^z\) 等。

**验证流程：**

1. 设定 d（如 5 或 8）和阈值 \(\varepsilon\)（如 \(10^{-3}\)）。
2. 对每个 \(N \ge d+2\) 计算该两点关联并打标签。
3. 模型检测 \(\Phi_3\)，得到“从某个 \(N_0\) 起永久衰减成立”。

---

## **公式 F4：Cluster 稳定子性质（稳定子态的全局安全性质）**

**LCL 形式：**

\[
\Phi_4 := G\,\ell_{\text{stab}}
\]

**物理背景与意义：**

1D cluster state 是稳定子态，满足局域稳定子生成元（例如 \(Z_{i-1}X_iZ_{i+1}\)）的期望为 +1。该公式检验：对所有链长 N（从一个最小长度起），稳定子关系都成立。

**标签建议：**

- \(\ell_{\text{stab}}\)：对所有适用的 i，
  \[
  \langle Z_{i-1}X_iZ_{i+1}\rangle \ge 1-\epsilon_{\text{stab}}
  \]

**验证流程：**

1. 对每个 N 计算所有局域稳定子期望值，打 \(\ell_{\text{stab}}\)。
2. 验证 \(G\,\ell_{\text{stab}}\)。
   这是一个很好的“强安全性质”案例：理想 cluster 会严格成立；加入噪声后会出现清晰失败点。

---

## **公式 F5：量子行走优势的“最终保持”（物理计算优势）**

**LCL 形式：**

\[
\Phi_5 := E\,G\,\ell_{\text{adv}}
\]

**物理背景与意义：**

离散时间量子行走相对经典随机行走在扩散速度/命中概率上有优势。该公式检验一种“优势最终出现并持续存在”的现象：存在一个步数 T，从此以后每一步都保持优势。

**标签建议（你可选其中一种）：**

- 扩散优势：\(\ell_{\text{adv}}\)：量子行走位置标准差 \(\sigma_q(t) \ge \alpha \sigma_c(t)\)
- 命中优势：\(\ell_{\text{adv}}\)：命中目标概率 \(p_q(t) \ge p_c(t) + \delta\)

**验证流程：**

1. 把第 t 步的态当作序列状态（对应 LCL 的“链长/时间步”维度）。
2. 每个 t 计算指标并与经典基线比较，打标签 \(\ell_{\text{adv}}\)。
3. 用模型检测验证 \(\Phi_5\)，得到“从某个 \(T_0\) 起优势永久成立”。
