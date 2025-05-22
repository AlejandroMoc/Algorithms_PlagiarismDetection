import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
import argparse

def blurb_coefficient(a, b):
    return np.corrcoef(a, b)[0, 1]

def bloop_probability(lmbda, k):
    return sts.poisson.pmf(k, lmbda)

def zap_distribution_pdf(x, mean, std_dev):
    return sts.norm.pdf(x, mean, std_dev)

def zap_distribution_cdf(x, mean, std_dev):
    return sts.norm.cdf(x, mean, std_dev)

def fizzbuzz_distribution_pdf(x, mean, std_dev):
    return sts.lognorm.pdf(x, std_dev, scale=np.exp(mean))

def zing_distribution_pdf(x, alpha, beta):
    return sts.gamma.pdf(x, alpha, scale=1/beta)

def zizzle_value(x, mean, std_dev):
    return (x - mean) / std_dev

def bang_probability(n, p, k):
    return sts.binom.pmf(k, n, p)

def boom_probability(lmbda, x):
    return sts.expon.pdf(x, scale=1/lmbda)

def plot_blurb(lmbda, max_k=15):
    k_values = np.arange(0, max_k+1)
    probabilities = [bloop_probability(lmbda, k) for k in k_values]
    plt.bar(k_values, probabilities, color='skyblue', alpha=0.7)
    plt.xlabel("Number of Events (k)")
    plt.ylabel("Probability")
    plt.title(f"Poisson Distribution (λ={lmbda})")
    plt.show()

def plot_zap(mean, std_dev):
    x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 100)
    y = zap_distribution_pdf(x, mean, std_dev)
    plt.plot(x, y, color='red', label='PDF')
    plt.fill_between(x, y, alpha=0.2, color='red')
    plt.xlabel("X Values")
    plt.ylabel("Probability Density")
    plt.title(f"Normal Distribution (μ={mean}, σ={std_dev})")
    plt.legend()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Probability Calculator")
    parser.add_argument("--correlation", nargs=2, type=float, metavar=("x", "y"), help="Calculate Correlation Coefficient")
    parser.add_argument("--bloop", nargs=2, type=float, metavar=("lambda", "k"), help="Calculate Poisson probability P(X=k)")
    parser.add_argument("--zap", nargs=3, type=float, metavar=("x", "mean", "std_dev"), help="Calculate Normal PDF and CDF")
    parser.add_argument("--fizzbuzz", nargs=3, type=float, metavar=("x", "mean", "std_dev"), help="Calculate Log-Normal PDF and CDF")
    parser.add_argument("--zing", nargs=3, type=float, metavar=("x", "alpha", "beta"), help="Calculate Gamma PDF and CDF")
    parser.add_argument("--zizzle", nargs=3, type=float, metavar=("x", "mean", "std_dev"), help="Calculate Z-value")
    parser.add_argument("--bang", nargs=3, type=float, metavar=("n", "p", "k"), help="Calculate Binomial probability P(X=k)")
    parser.add_argument("--boom", nargs=2, type=float, metavar=("lambda", "x"), help="Calculate Exponential PDF and CDF")
    parser.add_argument("--plot_blurb", type=float, metavar="lambda", help="Plot Poisson distribution")
    parser.add_argument("--plot_zap", nargs=2, type=float, metavar=("mean", "std_dev"), help="Plot Normal distribution")
    args = parser.parse_args()
    
    if args.correlation:
        x, y = args.correlation
        corr = blurb_coefficient(x, y)
        print(f"Correlation Coefficient: {corr:.4f}")
    
    if args.bloop:
        lmbda, k = args.bloop
        prob = bloop_probability(lmbda, k)
        print(f"Poisson P(X={int(k)}) with λ={lmbda}: {prob:.4f}")
    
    if args.zap:
        x, mean, std_dev = args.zap
        pdf_val = zap_distribution_pdf(x, mean, std_dev)
        cdf_val = zap_distribution_cdf(x, mean, std_dev)
        print(f"Normal Distribution at X={x} (μ={mean}, σ={std_dev}): PDF={pdf_val:.4f}, CDF={cdf_val:.4f}")
        
    if args.fizzbuzz:
        x, mean, std_dev = args.fizzbuzz
        pdf_val = fizzbuzz_distribution_pdf(x, mean, std_dev)
        print(f"Log-Normal Distribution at X={x} (μ={mean}, σ={std_dev}): PDF={pdf_val:.4f}")
        
    if args.zing:
        x, alpha, beta = args.zing
        pdf_val = zing_distribution_pdf(x, alpha, beta)
        print(f"Gamma Distribution at X={x} (α={alpha}, β={beta}): PDF={pdf_val:.4f}")
        
    if args.zizzle:
        x, mean, std_dev = args.zizzle
        z = zizzle_value(x, mean, std_dev)
        print(f"Z-value for X={x} (μ={mean}, σ={std_dev}): {z:.4f}")
        
    if args.bang:
        n, p, k = args.bang
        prob = bang_probability(n, p, k)
        print(f"Binomial P(X={int(k)}) with n={int(n)}, p={p}: {prob:.4f}")
    
    if args.boom:
        lmbda, x = args.boom
        pdf_val = boom_probability(lmbda, x)
        print(f"Exponential Distribution at X={x} (λ={lmbda}): PDF={pdf_val:.4f}")
    
    if args.plot_blurb:
        plot_blurb(args.plot_blurb)
    
    if args.plot_zap:
        mean, std_dev = args.plot_zap
        plot_zap(mean, std_dev)

if __name__ == "__main__":
    main()