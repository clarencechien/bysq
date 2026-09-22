"""v4 figures: §1 floor-cost vs floor-protection frontier.
python run_v4_figures.py -> results/v4_floor_frontier.png"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def main():
    rows = list(csv.DictReader(open(os.path.join(RES, "v4_floor.csv"))))
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5), sharey="row")
    colors = {"dynamic": "#1f77b4", "fixed": "#7f7f7f", "floor": "#d62728"}
    markers = {"S1": "o", "S2": "s", "S3": "^", "S4": "D"}
    for r_i, (age, pens) in enumerate([(65, "0.0"), (50, "0.0")]):
        for c_i, fr in enumerate(["0.35", "0.5", "0.7", "1.0"]):
            ax = axes[r_i, c_i]
            sub = [r for r in rows if r["engine"] == "US1934" and r["age"] == str(age)
                   and r["floor_ratio"] == fr and r["pension"] == pens and r["pension_mult"] == "1.0"]
            for r in sub:
                fam = r["family"]
                mk = markers.get(r["strategy"][:2], "o") if fam == "floor" else ("x" if fam == "fixed" else "P")
                ax.scatter(float(r["avg_med"]) * 100, float(r["breach1_pct"]), c=colors[fam], marker=mk, s=55,
                           alpha=0.85, edgecolors="k", linewidths=0.4)
                lab = r["strategy"].replace(" + VPW3 100/0", "").replace("(假設性)", "*")
                ax.annotate(lab, (float(r["avg_med"]) * 100, float(r["breach1_pct"])), fontsize=6,
                            xytext=(3, 2), textcoords="offset points")
            ax.set_title(f"retire at {age} / floor_ratio {fr} / no pension", fontsize=10)
            ax.set_xlabel("median real spending per alive year (% W0)", fontsize=8)
            if c_i == 0:
                ax.set_ylabel("P(breach floor while alive) %", fontsize=8)
            ax.grid(alpha=0.3)
    fig.suptitle("v4 S1: floor cost vs floor protection (US1934 stress sample, mortality, 4% spending). red = floor family, blue = dynamic rules, grey = fixed",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "v4_floor_frontier.png"), dpi=130)
    print("wrote results/v4_floor_frontier.png")


if __name__ == "__main__":
    main()
