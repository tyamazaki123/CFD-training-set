class SodExactSolution:
    def __init__(self, gamma=1.4):
        self.gamma = gamma

    def solve(self, x, t, left, right, x0=0.5):
        """
        Sodの解析解を返す
        x : 位置配列
        t : 時間
        left, right : [rho, u, p] 左右の初期値
        x0 : 初期の接触面
        return: rho, u, p, e_int
        """
        # 初期値分解
        rho_L, u_L, p_L = left
        rho_R, u_R, p_R = right
        g = self.gamma

        # 1. 真空判定と近似計算 (Sod解法)
        # ここではSciPyなど外部は使わず、ニュートン法を自前実装

        # 補助関数
        def f(p, rho, p0):
            if p > p0:
                A = 2/( (g+1)*rho )
                B = (g-1)/(g+1)*p0
                return (p - p0) * np.sqrt(A/(p + B))
            else:
                a = np.sqrt(g*p0/rho)
                return (2*a)/(g-1) * ( (p/p0)**((g-1)/(2*g)) - 1 )

        # ニュートン法で圧力を解く
        def phi(p):
            return f(p, rho_L, p_L) + f(p, rho_R, p_R) + (u_R-u_L)

        # 初期値と収束判定
        p_star = 0.5 * (p_L + p_R)
        for _ in range(100):
            fL = f(p_star, rho_L, p_L)
            fR = f(p_star, rho_R, p_R)
            dphi = (1/(rho_L*np.sqrt((g+1)/(2*g)*p_star/p_L + (g-1)/(2*g)))) + \
                   (1/(rho_R*np.sqrt((g+1)/(2*g)*p_star/p_R + (g-1)/(2*g))))
            dp = -phi(p_star) / (dphi if dphi != 0 else 1e-10)
            p_star += dp
            if abs(dp) < 1e-8:
                break

        # 接触面速度
        u_star = 0.5 * (u_L + u_R) + 0.5 * (f(p_star, rho_R, p_R) - f(p_star, rho_L, p_L))

        # 波速度
        a_L = np.sqrt(g*p_L/rho_L)
        a_R = np.sqrt(g*p_R/rho_R)
        S_L = u_L - a_L
        S_HL = u_star - np.sqrt(g*p_star/rho_L) if p_star > p_L else u_L - a_L * ((p_star/p_L)**((g-1)/(2*g)))
        S_HR = u_star + np.sqrt(g*p_star/rho_R) if p_star > p_R else u_R + a_R * ((p_star/p_R)**((g-1)/(2*g)))
        S_R = u_R + a_R

        # 結果配列
        rho = np.zeros_like(x)
        u   = np.zeros_like(x)
        p   = np.zeros_like(x)

        for i, xi in enumerate(x):
            s = (xi - x0)/t if t > 0 else 0
            if t == 0 or xi < x0:
                # 左領域
                rho[i] = rho_L
                u[i] = u_L
                p[i] = p_L
            elif s < S_L:
                rho[i] = rho_L
                u[i] = u_L
                p[i] = p_L
            elif s < S_HL:
                # 左側レアファクション波
                u[i] = 2/(g+1)*(a_L + (g-1)/2*u_L + s)
                a = a_L - (g-1)/2*(u[i] - u_L)
                p[i] = p_L * (a/a_L)**(2*g/(g-1))
                rho[i] = rho_L * (p[i]/p_L)**(1/g)
            elif s < u_star:
                # 左側接触不連続面
                rho[i] = rho_L * (p_star/p_L)**(1/g)
                u[i] = u_star
                p[i] = p_star
            elif s < S_HR:
                # 右側接触不連続面
                rho[i] = rho_R * (p_star/p_R)**(1/g)
                u[i] = u_star
                p[i] = p_star
            elif s < S_R:
                # 右側レアファクション波
                u[i] = 2/(g+1)*(-a_R + (g-1)/2*u_R + s)
                a = a_R + (g-1)/2*(u[i] - u_R)
                p[i] = p_R * (a/a_R)**(2*g/(g-1))
                rho[i] = rho_R * (p[i]/p_R)**(1/g)
            else:
                # 右領域
                rho[i] = rho_R
                u[i] = u_R
                p[i] = p_R

        e_int = p / ((g-1) * rho)
        return rho, u, p, e_int
