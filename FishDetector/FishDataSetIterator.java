/*
 *
 *  * ******************************************************************************
 *  *  * Copyright (c) 2019 Skymind AI Bhd.
 *  *  * Copyright (c) 2020 CertifAI Sdn. Bhd.
 *  *  *
 *  *  * This program and the accompanying materials are made available under the
 *  *  * terms of the Apache License, Version 2.0 which is available at
 *  *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *  *
 *  *  * Unless required by applicable law or agreed to in writing, software
 *  *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  *  * License for the specific language governing permissions and limitations
 *  *  * under the License.
 *  *  *
 *  *  * SPDX-License-Identifier: Apache-2.0
 *  *  *****************************************************************************
 *
 *
 */

package org.deeplearning4j.examples.DL4JProject.FishClassifier;

import org.apache.commons.io.FileUtils;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.examples.DL4JProject.DogBreedClassification.ArchiveUtils;
import org.deeplearning4j.examples.DL4JProject.Helper;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;

public class FishDataSetIterator {

    private static final Logger log = org.slf4j.LoggerFactory.getLogger(FishDataSetIterator.class);
    private static final int seed = 123;
    private static Random rng = new Random(seed);
    private static String dataDir;
    private static String downloadLink;
    private static Path trainDir, testDir;
    private static FileSplit trainData, testData;
    private static final int nChannels = 3;
    public static final int gridWidth = 13;
    public static final int gridHeight = 13;
    public static final int yolowidth = 416;
    public static final int yoloheight = 416;

    private static RecordReaderDataSetIterator makeIterator(InputSplit split, Path dir, int batchSize) throws Exception {

        ObjectDetectionRecordReader recordReader = new ObjectDetectionRecordReader(yoloheight, yolowidth, nChannels,
                gridHeight, gridWidth, new VocLabelProvider(dir.toString()));
        recordReader.initialize(split);
        RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, 1, true);
        iter.setPreProcessor(new ImagePreProcessingScaler(0, 1));

        return iter;
    }

    public static RecordReaderDataSetIterator trainIterator(int batchSize) throws Exception {
        return makeIterator(trainData, trainDir, batchSize);
    }

    public static RecordReaderDataSetIterator testIterator(int batchSize) throws Exception {
        return makeIterator(testData, testDir, batchSize);
    }

    public static void setup() throws IOException {
        log.info("Load data...");
        dataDir = Paths.get(
            System.getProperty("user.home"),
            Helper.getPropValues("dl4j_home.data")
        ).toString();
        trainDir = Paths.get(dataDir, "Fish Species Datasets","data2","train");
        testDir = Paths.get(dataDir, "Fish Species Datasets","data2","test");
        trainData = new FileSplit(new File(trainDir.toString()), NativeImageLoader.ALLOWED_FORMATS, rng);
        testData = new FileSplit(new File(testDir.toString()), NativeImageLoader.ALLOWED_FORMATS, rng);
    }


}
