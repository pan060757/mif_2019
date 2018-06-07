import datetime

start = '2006-06-01'
end = '2017-01-01'
datestart = datetime.datetime.strptime(start, '%Y-%m-%d')
dateend = datetime.datetime.strptime(end, '%Y-%m-%d')
datestart += datetime.timedelta(days=-1)
print(datestart.strftime('%Y%m%d'))