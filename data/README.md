### Sensor Training Data From Pittsburgh  
Obtained via the following `s5cmd` pattern
```
s5cmd --no-sign-request ls "s3://argoverse/datasets/av2/lidar/train/**/map/*_PIT_city_*"
```

To filter raw log id's
```
cat pit_lidar.log | awk '{print $4}' | awk -F'/' '{print $1}'
```

### Lidar Training Data From Pittsburgh
Obtained via the following `s5cmd` pattern
```
s5cmd --no-sign-request ls "s3://argoverse/datasets/av2/sensor/train/**/map/*_PIT_city_*"
```

To filter raw log id's
```
cat pit_sensor.log | awk '{print $4}' | awk -F'/' '{print $1}'
```
