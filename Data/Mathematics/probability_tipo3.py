import numpy as np
import scipy.stats as stats 
import matplotlib.pyplot as plt
import argparse

def coeff_correlacion(x, y):
    return np.corrcoef(x, y)[0, 1]

def probabilidad_poisson(lmbda, k):
    return stats.poisson.pmf(k, lmbda)

def pdf_distribucion_normal(x, mean, std_dev):
    return stats.norm.pdf(x, mean, std_dev)

def cdf_distribucion_normal(x, mean, std_dev):
    return stats.norm.cdf(x, mean, std_dev)

def pdf_distribucion_log_normal(x, mean, std_dev):
    return stats.lognorm.pdf(x, std_dev, scale=np.exp(mean))

def pdf_distribucion_gamma(x, alpha, beta):
    return stats.gamma.pdf(x, alpha, scale=1/beta)

def valor_z(x, mean, std_dev):
    return (x - mean) / std_dev

def probabilidad_binomial(n, p, k):
    return stats.binom.pmf(k, n, p)

def probabilidad_exponencial(lmbda, x):
    return stats.expon.pdf(x, scale=1/lmbda)

def graficar_poisson(lmbda, max_k=15):
    k_values = np.arange(0, max_k+1)
    probabilities = [probabilidad_poisson(lmbda, k) for k in k_values]
    plt.bar(k_values, probabilities, color='skyblue', alpha=0.7)
    plt.xlabel("Number of Events (k)")
    plt.ylabel("Probability")
    plt.title(f"Poisson Distribution (λ={lmbda})")
    plt.show()

def graficar_normal(mean, std_dev):
    x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 100)
    y = pdf_distribucion_normal(x, mean, std_dev)
    plt.plot(x, y, color='red', label='PDF')
    plt.fill_between(x, y, alpha=0.2, color='red')
    plt.xlabel("X Values")
    plt.ylabel("Probability Density")
    plt.title(f"Normal Distribution (μ={mean}, σ={std_dev})")
    plt.legend()
    plt.show()

def principal():
    parser = argparse.ArgumentParser(description="Probability Calculator")
    parser.add_argument("--correlation", nargs=2, type=float, metavar=("x", "y"), help="Calculate Correlation Coefficient")
    parser.add_argument("--poisson", nargs=2, type=float, metavar=("lambda", "k"), help="Calculate Poisson probability P(X=k)")
    parser.add_argument("--normal", nargs=3, type=float, metavar=("x", "mean", "std_dev"), help="Calculate Normal PDF and CDF")
    parser.add_argument("--lognormal", nargs=3, type=float, metavar=("x", "mean", "std_dev"), help="Calculate Log-Normal PDF and CDF")
    parser.add_argument("--gamma", nargs=3, type=float, metavar=("x", "alpha", "beta"), help="Calculate Gamma PDF and CDF")
    parser.add_argument("--zvalue", nargs=3, type=float, metavar=("x", "mean", "std_dev"), help="Calculate Z-value")
    parser.add_argument("--binomial", nargs=3, type=float, metavar=("n", "p", "k"), help="Calculate Binomial probability P(X=k)")
    parser.add_argument("--exponential", nargs=2, type=float, metavar=("lambda", "x"), help="Calculate Exponential PDF and CDF")
    parser.add_argument("--plot_poisson", type=float, metavar="lambda", help="Plot Poisson distribution")
    parser.add_argument("--plot_normal", nargs=2, type=float, metavar=("mean", "std_dev"), help="Plot Normal distribution")
    args = parser.parse_args()
    
    if args.correlation:
        x, y = args.correlation
        corr = coeff_correlacion(x, y)
        print(f"Correlation Coefficient: {corr:.4f}")
    
    if args.poisson:
        lmbda, k = args.poisson
        prob = probabilidad_poisson(lmbda, k)
        print(f"Poisson P(X={int(k)}) with λ={lmbda}: {prob:.4f}")
    
    if args.normal:
        x, mean, std_dev = args.normal
        pdf_val = pdf_distribucion_normal(x, mean, std_dev)
        cdf_val = cdf_distribucion_normal(x, mean, std_dev)
        print(f"Normal Distribution at X={x} (μ={mean}, σ={std_dev}): PDF={pdf_val:.4f}, CDF={cdf_val:.4f}")
        
    if args.lognormal:
        x, mean, std_dev = args.lognormal
        pdf_val = pdf_distribucion_log_normal(x, mean, std_dev)
        print(f"Log-Normal Distribution at X={x} (μ={mean}, σ={std_dev}): PDF={pdf_val:.4f}")
        
    if args.gamma:
        x, alpha, beta = args.gamma
        pdf_val = pdf_distribucion_gamma(x, alpha, beta)
        print(f"Gamma Distribution at X={x} (α={alpha}, β={beta}): PDF={pdf_val:.4f}")
        
    if args.zvalue:
        x, mean, std_dev = args.zvalue
        z = valor_z(x, mean, std_dev)
        print(f"Z-value for X={x} (μ={mean}, σ={std_dev}): {z:.4f}")
        
    if args.binomial:
        n, p, k = args.binomial
        prob = probabilidad_binomial(n, p, k)
        print(f"Binomial P(X={int(k)}) with n={int(n)}, p={p}: {prob:.4f}")
    
    if args.exponential:
        lmbda, x = args.exponential
        pdf_val = probabilidad_exponencial(lmbda, x)
        print(f"Exponential Distribution at X={x} (λ={lmbda}): PDF={pdf_val:.4f}")
    
    if args.plot_poisson:
        graficar_poisson(args.plot_poisson)
    
    if args.plot_normal:
        mean, std_dev = args.plot_normal
        graficar_normal(mean, std_dev)

if __name__ == "__main__":
    principal()