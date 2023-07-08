(test_model_target)=
# Test Model
Once the conda environment is [properly installed](install_pygem_target) and you have an understanding of the model components, you are ready to run the model. As described in the [model workflow](model_workflow_target), the model is meant to be run as a sequence of commands from the command line. To test if the model is properly installed and become familiar with the data, sample data for a test run are provided for Khumbu Glacier (RGI60-15.03733) [(download sample data)](https://drive.google.com/file/d/159zS-oGWLHG9nzkFdsf6Uh4-w9lJSt8H/view?usp=sharing). Below are two test workflows for the simple and advanced calibration schemes, and one additional calibrating and running simulations for marine-terminating ("tidewater") glaciers. Sample data for a test run are provided for Khumbu Glacier (RGI60-15.03733) [(download tidewater sample data)](https://drive.google.com/file/d/1Y9mVw9whEq7b4LURbOCxq-qopSwxoTnZ/view?usp=sharing).

```{toctree}
---
caption: Test Case:
maxdepth: 2
---

test_pygem_simple
test_pygem_advanced
test_pygem_tidewater
```

```{warning}
If your environment is not set up properly, errors will arise related to missing modules. We recommend that you work through adding the missing modules and use StackOverflow to identify any additional debugging issues related to potential missing modules or module dependencies.
```