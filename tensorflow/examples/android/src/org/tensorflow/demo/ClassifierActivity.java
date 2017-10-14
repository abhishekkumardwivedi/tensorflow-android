/*
 * Copyright 2016 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.demo;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.AsyncTask;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.Display;
import android.widget.Toast;

import com.google.api.client.extensions.android.http.AndroidHttp;
import com.google.api.client.googleapis.json.GoogleJsonResponseException;
import com.google.api.client.http.HttpTransport;
import com.google.api.client.json.JsonFactory;
import com.google.api.client.json.gson.GsonFactory;
import com.google.api.services.vision.v1.Vision;
import com.google.api.services.vision.v1.VisionRequest;
import com.google.api.services.vision.v1.VisionRequestInitializer;
import com.google.api.services.vision.v1.model.AnnotateImageRequest;
import com.google.api.services.vision.v1.model.BatchAnnotateImagesRequest;
import com.google.api.services.vision.v1.model.BatchAnnotateImagesResponse;
import com.google.api.services.vision.v1.model.EntityAnnotation;
import com.google.api.services.vision.v1.model.Feature;
import com.google.api.services.vision.v1.model.Image;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.Vector;
import org.tensorflow.demo.OverlayView.DrawCallback;
import org.tensorflow.demo.env.BorderedText;
import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.env.Logger;

public class ClassifierActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();
  private static final String TAG = ClassifierActivity.class.getSimpleName();

  protected static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final String CLOUD_VISION_API_KEY = "AIzaSyCCwJmcg7E0RfBSCBXaWLQynOur9-HxxXw";
  private static final String ANDROID_PACKAGE_HEADER = "X-Android-Package";
  private static final String ANDROID_CERT_HEADER = "X-Android-Cert";
  private static final int MAX_VISION_QUEUE = 5; // to reduce cloud API hit.
  private static int inCloudVisionQueue= 0;

  private ResultsView resultsView;

  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private long lastProcessingTimeMs;

  // These are the settings for the original v1 Inception model. If you want to
  // use a model that's been produced from the TensorFlow for Poets codelab,
  // you'll need to set IMAGE_SIZE = 299, IMAGE_MEAN = 128, IMAGE_STD = 128,
  // INPUT_NAME = "Mul", and OUTPUT_NAME = "final_result".
  // You'll also need to update the MODEL_FILE and LABEL_FILE paths to point to
  // the ones you produced.
  //
  // To use v3 Inception model, strip the DecodeJpeg Op from your retrained
  // model first:
  //
  // python strip_unused.py \
  // --input_graph=<retrained-pb-file> \
  // --output_graph=<your-stripped-pb-file> \
  // --input_node_names="Mul" \
  // --output_node_names="final_result" \
  // --input_binary=true
  private static final int INPUT_SIZE = 224;
  private static final int IMAGE_MEAN = 117;
  private static final float IMAGE_STD = 1;
  private static final String INPUT_NAME = "input";
  private static final String OUTPUT_NAME = "output";

  private static final String MODEL_FILE = "file:///android_asset/tensorflow_inception_graph.pb";
  private static final String LABEL_FILE =
      "file:///android_asset/imagenet_comp_graph_label_strings.txt";

  private static Set<String> codes = new HashSet<String>();

  private static final boolean MAINTAIN_ASPECT = true;

  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);


  private Integer sensorOrientation;
  private Classifier classifier;
  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;


  private BorderedText borderedText;


  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  private static final float TEXT_SIZE_DIP = 10;

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx = TypedValue.applyDimension(
        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    classifier =
        TensorFlowImageClassifier.create(
            getAssets(),
            MODEL_FILE,
            LABEL_FILE,
            INPUT_SIZE,
            IMAGE_MEAN,
            IMAGE_STD,
            INPUT_NAME,
            OUTPUT_NAME);

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    codes.addAll(Arrays.asList(new String[]
            {"AN","AP","AR","AS","BR","CG","CH","DD","DL","DN","GA","GJ","HR","HP","JH","JK","KA",
                    "KL","LD","MH","ML","MN","MP","MZ","NL","OD","PB","PY","RJ","SK","TN","TR","TS",
                    "UK","UP","WB" }));

    final Display display = getWindowManager().getDefaultDisplay();
    final int screenOrientation = display.getRotation();

    LOGGER.i("Sensor orientation: %d, Screen orientation: %d", rotation, screenOrientation);

    sensorOrientation = rotation + screenOrientation;

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Config.ARGB_8888);

    frameToCropTransform = ImageUtils.getTransformationMatrix(
        previewWidth, previewHeight,
        INPUT_SIZE, INPUT_SIZE,
        sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            renderDebug(canvas);
          }
        });
  }

  @Override
  protected void processImage() {
    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }
    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            final long startTime = SystemClock.uptimeMillis();
            final List<Classifier.Recognition> results = classifier.recognizeImage(croppedBitmap);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
            LOGGER.i("Detect: %s", results);
            for(int i=0; i < results.size(); i++) {
              String title = results.get(i).getTitle();
              Log.d(TAG, title);
              if((title.equals("car") || title.equals("sports car")) && inCloudVisionQueue < MAX_VISION_QUEUE) {
                Log.d(TAG,"Car found");
                cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                findLicensePlate(cropCopyBitmap);
              }
            }
            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            if (resultsView == null) {
              resultsView = (ResultsView) findViewById(R.id.results);
            }
            resultsView.setResults(results);
            requestRender();
            readyForNextImage();
          }
        });
  }

  public void findLicensePlate(Bitmap croppedBitmap) {
    if (croppedBitmap != null) {
      try {
        // scale the image to save on bandwidth
        Bitmap bitmap =
                scaleBitmapDown(croppedBitmap, 1200);
//                scaleBitmapDown(
//                        MediaStore.Images.Media.getBitmap(getContentResolver(), uri),
//                        1200);

        callCloudVision(bitmap);

      } catch (IOException e) {
        Log.d(TAG, "Image picking failed because " + e.getMessage());
      }
    } else {
      Log.d(TAG, "Image picker gave us a null image.");
    }
  }

  private void callCloudVision(final Bitmap bitmap) throws IOException {
    // Switch text to loading

    // Do the real work in an async task, because we need to use the network anyway
    new AsyncTask<Object, Void, String>() {
      @Override
      protected String doInBackground(Object... params) {
        try {
          HttpTransport httpTransport = AndroidHttp.newCompatibleTransport();
          JsonFactory jsonFactory = GsonFactory.getDefaultInstance();

          VisionRequestInitializer requestInitializer =
                  new VisionRequestInitializer(CLOUD_VISION_API_KEY) {
                    /**
                     * We override this so we can inject important identifying fields into the HTTP
                     * headers. This enables use of a restricted cloud platform API key.
                     */
                    @Override
                    protected void initializeVisionRequest(VisionRequest<?> visionRequest)
                            throws IOException {
                      super.initializeVisionRequest(visionRequest);

                      String packageName = getPackageName();
                      visionRequest.getRequestHeaders().set(ANDROID_PACKAGE_HEADER, packageName);

                      String sig = PackageManagerUtils.getSignature(getPackageManager(), packageName);

                      visionRequest.getRequestHeaders().set(ANDROID_CERT_HEADER, sig);
                    }
                  };

          Vision.Builder builder = new Vision.Builder(httpTransport, jsonFactory, null);
          builder.setVisionRequestInitializer(requestInitializer);

          Vision vision = builder.build();

          BatchAnnotateImagesRequest batchAnnotateImagesRequest =
                  new BatchAnnotateImagesRequest();
          batchAnnotateImagesRequest.setRequests(new ArrayList<AnnotateImageRequest>() {{
            AnnotateImageRequest annotateImageRequest = new AnnotateImageRequest();

            // Add the image
            Image base64EncodedImage = new Image();
            // Convert the bitmap to a JPEG
            // Just in case it's a format that Android understands but Cloud Vision
            ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
            bitmap.compress(Bitmap.CompressFormat.JPEG, 90, byteArrayOutputStream);
            byte[] imageBytes = byteArrayOutputStream.toByteArray();

            // Base64 encode the JPEG
            base64EncodedImage.encodeContent(imageBytes);
            annotateImageRequest.setImage(base64EncodedImage);

            // add the features we want
            annotateImageRequest.setFeatures(new ArrayList<Feature>() {{
              Feature labelDetection = new Feature();
              labelDetection.setType("TEXT_DETECTION");
              labelDetection.setMaxResults(10);
              add(labelDetection);
            }});

            // Add the list of one thing to the request
            add(annotateImageRequest);
          }});

          Vision.Images.Annotate annotateRequest =
                  vision.images().annotate(batchAnnotateImagesRequest);
          // Due to a bug: requests to Vision API containing large images fail when GZipped.
          annotateRequest.setDisableGZipContent(true);
          Log.d(TAG, "created Cloud Vision request object, sending request");

          BatchAnnotateImagesResponse response = annotateRequest.execute();
          inCloudVisionQueue++;
          return convertResponseToString(response);

        } catch (GoogleJsonResponseException e) {
          Log.d(TAG, "failed to make API request because " + e.getContent());
        } catch (IOException e) {
          Log.d(TAG, "failed to make API request because of other IOException " +
                  e.getMessage());
        }
        return "Cloud Vision API request failed. Check logs for details.";
      }

      protected void onPostExecute(String result) {
        if(result != null) {
          Toast.makeText(getApplicationContext(), result, Toast.LENGTH_LONG).show();

        }
        Log.d(TAG, "----------------------------------------");
        Log.d(TAG, "result:" + result);
        Log.d(TAG, "----------------------------------------");

      }
    }.execute();
  }

  public Bitmap scaleBitmapDown(Bitmap bitmap, int maxDimension) {

    int originalWidth = bitmap.getWidth();
    int originalHeight = bitmap.getHeight();
    int resizedWidth = maxDimension;
    int resizedHeight = maxDimension;

    if (originalHeight > originalWidth) {
      resizedHeight = maxDimension;
      resizedWidth = (int) (resizedHeight * (float) originalWidth / (float) originalHeight);
    } else if (originalWidth > originalHeight) {
      resizedWidth = maxDimension;
      resizedHeight = (int) (resizedWidth * (float) originalHeight / (float) originalWidth);
    } else if (originalHeight == originalWidth) {
      resizedHeight = maxDimension;
      resizedWidth = maxDimension;
    }
    return Bitmap.createScaledBitmap(bitmap, resizedWidth, resizedHeight, false);
  }

  private String convertResponseToString(BatchAnnotateImagesResponse response) {
    String message = "";
    String number = null;
    --inCloudVisionQueue; // no need to worry about synchronization. It won't harm.

    List<EntityAnnotation> labels = response.getResponses().get(0).getTextAnnotations();
    if (labels != null) {
      for (EntityAnnotation label : labels) {
        message += String.format(Locale.US, "%.3f: %s", label.getScore(), label.getDescription());
        number = parseLicensePlate(message);
        if(number!=null) {
          return number;
        }
        message += " ";
      }
    } else {
      Log.d(TAG,"license plate couldn't be recognized");
      return null;
    }
    return parseLicensePlate(message);
  }

  String parseLicensePlate(String message) {
    Log.d(TAG, "Message: " + message);
    message = message.replaceAll("\\s","");
    Log.d(TAG, "Formatted Message: " + message);

    Iterator iterator = codes.iterator();

    // if string qualifies to be license plate in India.
    // this will be removed over the time to put logic at ML to qualify
    while (iterator.hasNext()){
      int index = message.indexOf(iterator.next().toString());
      if(index > -1 && (message.length() >= (index+9))) {
        String number = message.substring(index, index + 9);
        char[] num = number.toCharArray();

        if (Character.isDigit(num[2])
                && Character.isDigit(num[3])
                && Character.isAlphabetic(num[4])
  //              && Character.isDigit(num[5])
  //              && Character.isDigit(num[6])
  //              && Character.isDigit(num[7])
  //              && Character.isDigit(num[8])
                ) {
          Log.d(TAG, "Number: " + number);

          return number;
        }
      }
    }
    return null;
  }

  @Override
  public void onSetDebug(boolean debug) {
    classifier.enableStatLogging(debug);
  }

  private void renderDebug(final Canvas canvas) {
    if (!isDebug()) {
      return;
    }
    final Bitmap copy = cropCopyBitmap;
    if (copy != null) {
      final Matrix matrix = new Matrix();
      final float scaleFactor = 2;
      matrix.postScale(scaleFactor, scaleFactor);
      matrix.postTranslate(
          canvas.getWidth() - copy.getWidth() * scaleFactor,
          canvas.getHeight() - copy.getHeight() * scaleFactor);
      canvas.drawBitmap(copy, matrix, new Paint());

      final Vector<String> lines = new Vector<String>();
      if (classifier != null) {
        String statString = classifier.getStatString();
        String[] statLines = statString.split("\n");
        for (String line : statLines) {
          lines.add(line);
        }
      }

      lines.add("Frame: " + previewWidth + "x" + previewHeight);
      lines.add("Crop: " + copy.getWidth() + "x" + copy.getHeight());
      lines.add("View: " + canvas.getWidth() + "x" + canvas.getHeight());
      lines.add("Rotation: " + sensorOrientation);
      lines.add("Inference time: " + lastProcessingTimeMs + "ms");

      borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
    }
  }
}
