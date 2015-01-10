/**
 * 
 */

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Jiachen
 *
 */
public class featGen {

    /**
     * @param args
     */
    public static void main(String[] args) throws Exception {
        // develop your routine here
        if (args.length != 1) {
            System.out.println("Usage: <input feature file>");
            return;
        }
        
        sampleRoutine(args[0]);

    }

    public static void sampleRoutine(String inputfile) throws IOException {

        System.out.println("input file: " + inputfile);
        System.out.println("output feature file: " + inputfile + ".out");

        BufferedReader br = new BufferedReader(new InputStreamReader(
                new FileInputStream(inputfile)));
        BufferedWriter bw = new BufferedWriter(new FileWriter(inputfile
                + ".feat.out"));
        BufferedWriter bw_label = new BufferedWriter(new FileWriter(inputfile
                + ".label.out"));

        // set feature type here, each sample will be stored in an list
        List<Double> featList = new ArrayList<>();
        String format = "ssv";
        int featDim = -1;

        // load original feature file
        String line = null;
        int ctAll = 0, ctValid = 0;
        while ((line = br.readLine()) != null) {
            if (line.length() == 0) {
                continue;
            }
            ctAll++;

            // process one data sample
            line = line.trim();
            String[] feats = line.split(" ");

            if (featDim == -1) {
                // suppose the first sample is right, use its
                // dimension as the required feature dimension
                featDim = feats.length;
            }

            // check the feature dimension
            if (feats.length != featDim) {
                System.out.println("invalid sample: " + line);
                continue;
            }
            ctValid++;

            // add to featList for converting format
            for (int i = 1; i < feats.length; i++) {
                featList.add(Double.valueOf(feats[i]));
            }

            // write out feature & label to file
            String sample = toFormatedString(featList, format);
            String label = (int) Float.parseFloat(feats[0]) + "\n";
            bw.write(sample);
            bw_label.write(label);

            featList.clear();
        }

        // clean up
        br.close();
        bw.close();
        bw_label.close();

        // report
        System.out.println("Process Finised!");
        System.out.println("Number of lines: " + ctAll);
        System.out.println("Valid lines: " + ctValid + ", invalid lines: "
                + (ctAll - ctValid));
    }

    // To space separated value file
    private static String toFormatedString(List<Double> list, String format) {
        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < list.size() - 1; i++) {
            sb.append(list.get(i));
            if ("tsv".equals(format)) {
                sb.append("\t");
            } else if ("csv".equals(format)) {
                sb.append(",");
            } else {
                sb.append(" ");
            }
        }
        sb.append(list.get(list.size() - 1));
        sb.append("\n");

        return sb.toString();
    }

}
