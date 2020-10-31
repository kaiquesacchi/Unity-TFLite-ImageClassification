using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TensorFlowLite;

public class ImageClassificationScript : MonoBehaviour
{
    [SerializeField] RawImage cameraDisplay;
    [SerializeField] Text outputTextDisplay;
    [SerializeField] string fileName = "hand_gesture_model.tflite";

    private WebCamDevice[] devices;
    private WebCamDevice chosenCamera;
    private WebCamTexture cameraTexture;

    private Interpreter interpreter;
    private float[,] inputs = new float[128, 128];
    private float[] outputs = new float[2];
    private bool isProcessing;

    void Start()
    {
        StartCamera();
        outputTextDisplay.text = "Camera Inicializada!";
        StartInterpreter();
        outputTextDisplay.text = "Interpretador Inicializado!";
    }

    void Update()
    {
        if (!isProcessing && cameraTexture)
        {
            Invoke(cameraTexture);
        }
    }

    void OnDestroy()
    {
        interpreter?.Dispose();
    }

    void StartCamera()
    {
        devices = WebCamTexture.devices;
        chosenCamera = devices[0];
        foreach (var device in devices)
        {
            if (device.isFrontFacing) chosenCamera = device;
        }
        cameraTexture = new WebCamTexture(chosenCamera.name, 128, 128, 60);
        cameraTexture.Play();
        cameraDisplay.texture = cameraTexture;
    }

    void StartInterpreter()
    {
        
        var options = new InterpreterOptions()
        {
            threads = 2,
            useNNAPI = false,
        };
        outputTextDisplay.text = "AQUI";
        interpreter = new Interpreter(FileUtil.LoadFile(fileName), options);
        interpreter.ResizeInputTensor(0, new int[] { 1, 128, 128, 1 });
        interpreter.AllocateTensors();
    }

    void Invoke(WebCamTexture texture)
    {
        isProcessing = true;

        Color[] pixels = texture.GetPixels();
        for (int i = 0; i < 128; i++)
        {
            for (int j = 0; j < 128; j++)
            {
                int W = (int) (texture.width * ((float)j / 128));
                int H = (int) (texture.height * ((float)i / 128));
                inputs[i, j] = pixels[H * texture.width + W].grayscale;
            }
        }

        float startTime = Time.realtimeSinceStartup;
        interpreter.SetInputTensorData(0, inputs);
        interpreter.Invoke();
        interpreter.GetOutputTensorData(0, outputs);
        float duration = Time.realtimeSinceStartup - startTime;

        if (outputs[0] > 0.5)
        {
            outputTextDisplay.text = "FECHADA: " + (outputs[0] * 100).ToString() + "%";
        }
        else
        {
            outputTextDisplay.text = "ABERTA: " + ((1 - outputs[0]) * 100).ToString() + "%";
        }
        isProcessing = false;
    }
}
