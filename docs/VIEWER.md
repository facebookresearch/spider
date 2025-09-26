```
et -t 9876:9876 fair-sc
lsof -i :9090
kill -9 any process already in use
rerun --serve --port 9876
# alternatively
rerun datapoints --serve --port 9876
```

```
rerun --connect rerun+http://localhost:9876/proxy
```