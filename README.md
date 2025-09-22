# Ricci Flow on Neural Network Fisher Manifolds: Application to Image Denoising

## 1. Mathematical Foundation

### 1.1 The Fisher Information Manifold

Consider a neural network as a map $f: \mathbb{R}^n \times \Theta \to \mathcal{P}(\mathbb{R}^m)$ where $\Theta$ is the parameter space and $\mathcal{P}$ denotes probability distributions. The Fisher information metric on $\Theta$ is:

$$g_{ij}(\theta) = \mathbb{E}_{p(x,y|\theta)}\left[\frac{\partial \log p(y|x;\theta)}{\partial \theta_i} \cdot \frac{\partial \log p(y|x;\theta)}{\partial \theta_j}\right]$$

This endows the parameter space with a Riemannian structure $(Θ, g)$.

### 1.2 Ricci Curvature and Information Geometry

The Ricci tensor $R_{ij}$ measures the deviation of the manifold from flat space. In information geometry:

$$R_{ij} = \partial_k \Gamma^k_{ij} - \partial_j \Gamma^k_{ik} + \Gamma^k_{kl}\Gamma^l_{ij} - \Gamma^k_{jl}\Gamma^l_{ik}$$

where $\Gamma^k_{ij}$ are the Christoffel symbols of the Fisher metric.

For a statistical manifold with Fisher-Rao metric $g_{ij}$, the Ricci curvature tensor is:

$$\text{Ric}_{ij} = -\frac{1}{2} \frac{\partial^2}{\partial \theta^k \partial \theta^l} \log \det(g) \cdot g_{ij} + \text{lower order terms}$$



### 1.3 The Ricci Flow Evolution

The normalized Ricci flow evolves the metric according to:

$$\frac{\partial g_{ij}}{\partial \tau} = -2\left(R_{ij} - \frac{r}{2n}g_{ij}\right)$$

where $r = g^{ij}R_{ij}$ is the scalar curvature and $n = \dim(\Theta)$.

Let $p_\theta(t)$ be a family of distributions evolving according to the Fisher-Rao Ricci flow:

$$
\frac{\partial \theta^i}{\partial t} = -g^{ij} \, \text{Ric}_{jk} \, g^{kl} \, \theta^l
$$

Then the Fisher information decreases monotonically along the flow.




---

## 2. Noise as Curvature Perturbation

### 2.1 The Geometric Signature of Noise

**Hypothesis**: In the probability flow through a neural network, noise manifests as isolated pockets of high curvature gradient.

Consider the flow map $\Phi^{(l)}: \mathcal{P}^{(l)} \to \mathcal{P}^{(l+1)}$ between layers. For clean images:
$$\text{Ric}(\Phi_{\text{clean}}) \approx \text{const}$$

For noisy images:
$$\text{Ric}(\Phi_{\text{noisy}}) = \text{Ric}(\Phi_{\text{clean}}) + \epsilon \cdot \delta_{\text{noise}}$$

where $\delta_{\text{noise}}$ represents localized curvature spikes.

Let $p_{\text{clean}}$ be the distribution of a clean image patch and 

$$
p_{\text{noisy}} = (1 - \epsilon) p_{\text{clean}} + \epsilon q
$$

be its noisy version with contamination \( q \). The Fisher-Rao Ricci curvature satisfies:

$$
|\text{Ric}(p_{\text{noisy}})| \geq |\text{Ric}(p_{\text{clean}})| + O(\epsilon)
$$

with equality only when $q = p_{\text{clean}}$.



## 3. Discrete Approximation via Ollivier-Ricci Curvature

### 3.1 From Continuous to Discrete

We approximate the continuous Fisher manifold with a graph $G = (V, E)$ where:
- Vertices: Image patches encoded as probability distributions
- Edges: Weighted by distribution similarity

The Ollivier-Ricci curvature for edge $(i,j)$:

$$\kappa(i,j) = 1 - \frac{W_1(\mu_i, \mu_j)}{d_G(i,j)}$$

where $W_1$ is the Wasserstein-1 distance between neighborhood measures.

For our application, we define the probability measure as:
$$\mu_i(k) = \frac{w_{ik}}{\sum_{l \in N(i)} w_{il}}$$


Under suitable conditions, as the graph becomes dense:
$$\lim_{|V| \to \infty} \kappa_{\text{Ollivier}} \to \text{Ric}_{\text{continuous}}$$

---

## 4. Experimental Results

### 4.1 Training Dynamics

The model was trained with loss:
$$\mathcal{L} = \|\text{reconstruction} - \text{target}\|^2 + \beta \cdot \|\kappa_{\text{predicted}} - \kappa_{\text{computed}}\|^2$$

**Observation**: The system rapidly converges to uniform curvature $\kappa \approx 0.983 \pm 0.006$ across all edges.

### 4.2 Denoising Performance

| Noise σ | Input PSNR | Output PSNR | Δ PSNR |
|---------|------------|-------------|---------|
| 15      | 24.80 dB   | 16.71 dB    | -8.09 dB |
| 25      | 20.51 dB   | 16.99 dB    | -3.52 dB |
| 35      | 17.79 dB   | 17.34 dB    | -0.44 dB |
| 50      | 15.02 dB   | 17.89 dB    | +2.87 dB |

### 4.3 Curvature Distribution Analysis

The predicted curvature distribution exhibits:
- Range: [0.958, 0.991]
- Mean: 0.983
- Std: 0.006
- Correlation with edge weights: $R^2 = 0.95$

This extreme concentration suggests the model has collapsed to a geometrically trivial solution.

---

## 5. Why did it fail

### 5.1 The Discretization Bottleneck

The encoding into 64-dimensional probability distributions creates severe information loss:

$$I(X; T) \ll I(X; Y)$$

where $X$ = input patches, $T$ = discrete distributions, $Y$ = clean patches.

By the data processing inequality:
$$I(T; Y) \leq \min\{I(X; T), I(T; Y)\}$$

The bottleneck $\dim(T) = 64 \ll \dim(X) = 3072$ fundamentally limits reconstruction quality.

### 5.2 Uniform Curvature as Trivial Geometry

The convergence to constant curvature indicates the system has evolved to:
$$\lim_{t \to \infty} (M, g_t) \approx S^n/\Gamma$$

a quotient of the sphere - the maximally symmetric space. This represents complete loss of geometric structure needed for denoising.

### 5.3 Scale Mismatch

The hypothesis that "noise creates isolated high-curvature regions" assumes:
1. Local perturbations in pixel space → Local perturbations in probability space
2. These perturbations are detectable at patch scale (32×32)

However, the discretization and pooling operations destroy this correspondence.

---


## 6. Conclusion

Noise in images often manifests as localized perturbations at the pixel level or in small image patches. The Fisher manifold of a neural network's parameters doesn't capture this local geometry, and thus Ricci flow on the Fisher manifold smooths over these noise features, leading to a loss of important image structure. Instead of using the Fisher manifold, we can apply Ricci flow directly to the image space, where noise is present. By applying Ricci flow to the image space itself, we could more effectively reduce noise without losing key features, making it a more suitable approach for denoising tasks than working with the Fisher manifold of neural network parameters.

