# filename: stock_analysis.py
import yfinance as yf
import numpy as np

# 삼성전자 주식 데이터 가져오기
ticker = '005930.KS'  # 삼성전자의 티커
data = yf.download(ticker, period='3mo')

# 종가 데이터 추출
closing_prices = data['Close']

# 평균과 분산 계산
average_price = np.mean(closing_prices)
variance_price = np.var(closing_prices)

# 결과 출력
print(f"삼성전자 최근 3개월 평균 주가: {average_price:.2f} 원")
print(f"삼성전자 최근 3개월 주가 분산: {variance_price:.2f} 원²")