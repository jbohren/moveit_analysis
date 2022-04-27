
Robot Analysis with MoveIt
==========================

## Usage

### Analyze a Serial Chain Manipulator

Several metrics can be computed for a serial chain manipulator to inform how it
can be used, and what it's capabilities are.

```
ros2 run analyze_chain.py --params-file params.yaml
```

params.yaml:
```yaml
chain_analyzer:
  ros__parameters:
    base_link: ""
    tip_link:  ""
    group:     ""
    cartesian_sample_space:
      step: 0.06
      range:
        x: [0.5, 1.0]
        y: [0.0, 1.0]
        z: [0.0, 1.0]
```

