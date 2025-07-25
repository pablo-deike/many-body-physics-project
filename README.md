# Measurement induced phase transitions

## Course: Computational many body methods physics

### How to run

```bash
pip install -r requirements.txt
```

There are two main scripts to run the code:

- `site_transition.py`: The script runs the different entanglement entropy calculations per reduced system.
- `time_evol.py`: The script runs the time evolution of the system.
  Both of these will generate datasets that are used for the `plotting.ipynb` notebook, so you need to run them both first (warning: they are slow so get a coffe in the meantime). In this notebook, you can visualize the results of the calculations.
