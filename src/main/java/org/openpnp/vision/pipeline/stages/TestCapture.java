package org.openpnp.vision.pipeline.stages;

import org.json.JSONObject;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.openpnp.machine.reference.camera.OpenPnpCaptureCamera;
import org.openpnp.model.Configuration;
import org.openpnp.model.Location;
import org.openpnp.model.Part;
import org.openpnp.spi.Actuator;
import org.openpnp.spi.Camera;
import org.openpnp.spi.Nozzle;
import org.openpnp.util.OpenCvUtils;
import org.openpnp.vision.FluentCv.ColorSpace;
import org.openpnp.vision.pipeline.*;
import org.openpnp.vision.pipeline.ui.PipelinePropertySheetTable;
import org.simpleframework.xml.Attribute;
import org.simpleframework.xml.Element;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;


@Stage(
        category = "Image Processing",
        description = "Capture an image from the pipeline camera.")

public class TestCapture extends CvStage {
    @Attribute(required = false)
    @Property(description = "Use the default camera lighting.")
    private boolean defaultLight = false;

    @Element(required = false)
    @Property(description = "Light actuator value or profile, if default camera lighting is disabled.")
    private Object light = null;

    @Attribute
    @Property(description = "Wait for the camera to settle before capturing an image.")
    private boolean settleFirst;

    @Attribute(required = false)
    @Property(description = "Number of camera images to average.")
    private int count = 1;

    public boolean isDefaultLight() {
        return defaultLight;
    }

    public void setDefaultLight(boolean defaultLight) {
        this.defaultLight = defaultLight;
    }

    public Object getLight() {
        return light;
    }

    public void setLight(Object light) {
        this.light = light;
    }

    public boolean isSettleFirst() {
        return settleFirst;
    }

    public void setSettleFirst(boolean settleFirst) {
        this.settleFirst = settleFirst;
    }

    public int getCount() {
        return count;
    }

    public void setCount(int count) {
        if (count > 0) {
            this.count = count;
        } else {
            this.count = 1;
        }
    }

    @Override
    public Result process(CvPipeline pipeline) throws Exception {
        OpenPnpCaptureCamera camera = (OpenPnpCaptureCamera) pipeline.getProperty("camera");
        boolean cameraAutoOrigin = camera.getExposure().isAuto();
        int cameraExposureOrigin = camera.getExposure().getValue();

        Nozzle nozzle = (Nozzle) pipeline.getProperty("nozzle");
        Actuator actuator = (Actuator) pipeline.getProperty("lightActuator");

        Location baseLocation = nozzle.getLocation();

        Part part = nozzle.getPart();

        if (camera == null) {
            throw new Exception("No Camera set on pipeline.");
        }

        if (nozzle == null) {
            throw new Exception("No Nozzle set on pipeline.");
        }

        try {
            // Light, settle and capture the image. Keep the lights on for possible averaging.
//            camera.actuateLightBeforeCapture((defaultLight ? null : getLight()));
            Integer captureCount = 5;
            float maxXDeviation = 1;
            float maxYDeviation = 1;
            float maxRDeviation = 15;

            Integer lightCount = 20;
            float lightMaxValue = 8;
            int exposureMaxValue = -7;
            int exposureMixValue = -11;
            for (int exposure = exposureMixValue; exposure < exposureMaxValue; exposure++) {
                camera.getExposure().setAuto(false);
                camera.getExposure().setValue(exposure);
                camera.open();
                for (Integer light = 3; light <= lightCount; light++)
                    for (Integer capture = 0; capture < captureCount; capture++) {
                        float curXDeviation = (float) (new Random().nextFloat() - 0.5) * 2 * maxXDeviation;
                        float curYDeviation = (float) (new Random().nextFloat() - 0.5) * 2 * maxYDeviation;
                        float curRDeviation = (float) (new Random().nextFloat() - 0.5) * 2 * maxRDeviation;

                        Location offsetLocation = new Location(baseLocation.getUnits(), curXDeviation, curYDeviation, 0, curRDeviation);

                        nozzle.moveTo(baseLocation.addWithRotation(offsetLocation));

                        float curLight = (1 << light) -1;
                        actuator.actuate(curLight);

                        try {
                            settleFirst = true;
                            BufferedImage bufferedImage = (settleFirst ? camera.settleAndCapture() : camera.capture());
                            Mat image = OpenCvUtils.toMat(bufferedImage);
                            if (count > 1) {
                                // Perform averaging in channel type double.
                                image.convertTo(image, CvType.CV_64F);
                                Mat avgImage = image;
                                double beta = 1.0 / count;
                                Core.addWeighted(avgImage, 0, image, beta, 0, avgImage); // avgImage = image/count
                                for (int i = 1; i < count; i++) {
                                    image = OpenCvUtils.toMat(camera.capture());
                                    image.convertTo(image, CvType.CV_64F);
                                    Core.addWeighted(avgImage, 1, image, beta, 0, avgImage); // avgImage = avgImag + image/count
                                    // Release the additional image.
                                    image.release();
                                }
                                avgImage.convertTo(avgImage, CvType.CV_8U);
                                image = avgImage;
                            }

                            File file = Configuration.get().createResourceFile(getClass(), "TestImage", capture.toString() + ".jpeg");
                            Imgcodecs.imwrite(file.getAbsolutePath(), image);

                            JSONObject jsonPart = new JSONObject();
                            jsonPart.put("name", part.getName());
                            jsonPart.put("size", Arrays.asList(part.getPackage().getFootprint().getBodyHeight(), part.getPackage().getFootprint().getBodyHeight()));

                            JSONObject jsonObject = new JSONObject();
                            jsonObject.put("part", part);
                            jsonObject.put("XDeviation", curXDeviation);
                            jsonObject.put("YDeviation", curYDeviation);
                            jsonObject.put("RDeviation", curRDeviation);
                            jsonObject.put("light", curLight);
                            jsonObject.put("part", jsonPart);
                            jsonObject.put("unitsPerPixel", Arrays.asList(camera.getUnitsPerPixel().getX(), camera.getUnitsPerPixel().getY()));

                            FileWriter jsonFile = new FileWriter(file.getAbsolutePath().replace(".jpeg", ".json"));
                            jsonFile.write(jsonObject.toString());
                            jsonFile.close();
                        } finally {
                            // Always switch off the light.
                            camera.actuateLightAfterCapture();
                        }
                    }
            }
        } catch (Exception e) {
            // These machine exceptions are terminal to the pipeline.
            throw new TerminalException(e);
        } finally {
            camera.getExposure().setAuto(cameraAutoOrigin);
            camera.getExposure().setValue(cameraExposureOrigin);
            camera.open();

            nozzle.moveTo(baseLocation);
        }

        return null;
    }

    @Override
    public void customizePropertySheet(PipelinePropertySheetTable table, CvPipeline pipeline) {
        super.customizePropertySheet(table, pipeline);
        Camera camera = (Camera) pipeline.getProperty("camera");
        if (camera != null) {
            Actuator actuator = camera.getLightActuator();
            String propertyName = "light";
            table.customizeActuatorProperty(propertyName, actuator);
        }
    }
}
