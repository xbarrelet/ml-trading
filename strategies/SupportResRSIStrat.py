from backtesting import Strategy


class SupportResRSIStrat(Strategy):
    n1 = 8
    n2 = 6
    backCandles = 140
    counter = backCandles + n1

    def init(self):
        self.signal = check_candle_signal(self.counter, self.n1, self.n2, self.backCandles, self.data)

    def next(self):
        if self.signal == 1:
            self.position.close()
            self.buy()

        if self.signal == 2:
            self.position.close()
            self.sell()

        # if self.signal == 0:
        #     self.position.close()

        self.counter += 1


wick_threshold = 0.0001


def support(df1, l, n1, n2):  # n1 and n2 represent the number of candles before and after candle l
    if l - n1 < 0 or l + n2 >= len(df1):
        return 0

    if (df1.Low[l - n1:l].min() < df1.Low[l] or
            df1.Low[l + 1:l + n2 + 1].min() < df1.Low[l]):
        return 0

    candle_body = abs(df1.Open[l] - df1.Close[l])
    lower_wick = min(df1.Open[l], df1.Close[l]) - df1.Low[l]
    if (lower_wick > candle_body) and (lower_wick > wick_threshold):
        return 1

    return 0


def resistance(df1, l, n1, n2):  # n1 and n2 represent the number of candles before and after candle l
    if l - n1 < 0 or l + n2 >= len(df1):
        return 0

    if (df1.High[l - n1:l].max() > df1.High[l] or
            df1.High[l + 1:l + n2 + 1].max() > df1.High[l]):
        return 0

    candle_body = abs(df1.Open[l] - df1.Close[l])
    upper_wick = df1.High[l] - max(df1.Open[l], df1.Close[l])
    if (upper_wick > candle_body) and (upper_wick > wick_threshold):
        return 1

    return 0


def closeResistance(l, levels, lim, df):
    if len(levels) == 0:
        return 0
    nearest_level = min(levels, key=lambda x: abs(x - df.High[l]))
    c1 = abs(df.High[l] - nearest_level) <= lim
    c2 = abs(max(df.Open[l], df.Close[l]) - nearest_level) <= lim
    c3 = min(df.Open[l], df.Close[l]) < nearest_level
    c4 = df.Low[l] < nearest_level
    if (c1 or c2) and c3 and c4:
        return nearest_level
    else:
        return 0


def closeSupport(l, levels, lim, df):
    if len(levels) == 0:
        return 0
    nearest_level = min(levels, key=lambda x: abs(x - df.Low[l]))
    c1 = abs(df.Low[l] - nearest_level) <= lim
    c2 = abs(min(df.Open[l], df.Close[l]) - nearest_level) <= lim
    c3 = max(df.Open[l], df.Close[l]) > nearest_level
    c4 = df.High[l] > nearest_level
    if (c1 or c2) and c3 and c4:
        return nearest_level
    else:
        return 0


def is_below_resistance(l, level_backCandles, level, df):
    return df.loc[l - level_backCandles:l - 1, 'high'].max() < level


def is_above_support(l, level_backCandles, level, df):
    return df.loc[l - level_backCandles:l - 1, 'low'].min() > level


def check_candle_signal(l, n1, n2, backCandles, df):
    ss = []
    rr = []
    for subrow in range(l - backCandles, l - n2):
        if support(df, subrow, n1, n2):
            ss.append(df.Low[subrow])
        if resistance(df, subrow, n1, n2):
            rr.append(df.High[subrow])

    # Remove close distance support levels
    ss.sort()  # Keep lowest support when popping a level
    for i in range(1, len(ss)):
        if i >= len(ss):
            break
        if abs(ss[i] - ss[i - 1]) <= 0.0001:
            ss.pop(i)

    # Remove close distance resistance levels
    rr.sort(reverse=True)  # Keep highest resistance when popping one
    for i in range(1, len(rr)):
        if i >= len(rr):
            break
        if abs(rr[i] - rr[i - 1]) <= 0.0001:
            rr.pop(i)

    # Join support and resistance levels and merge close ones
    rrss = rr + ss
    rrss.sort()
    for i in range(1, len(rrss)):
        if i >= len(rrss):
            break
        if abs(rrss[i] - rrss[i - 1]) <= 0.0001:
            rrss.pop(i)

    cR = closeResistance(l, rrss, 150e-5, df)
    cS = closeSupport(l, rrss, 150e-5, df)

    # Determine trading signal
    if cR and is_below_resistance(l, 6, cR, df) and df.RSI[l - 1:l].min() < 45:  # and df.RSI[l] > 65
        return 1
    elif cS and is_above_support(l, 6, cS, df) and df.RSI[l - 1:l].max() > 55:  # and df.RSI[l] < 35
        return 2
    else:
        return 0
