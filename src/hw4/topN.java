import java.io.IOException;
import java.util.*;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.*;
import org.apache.hadoop.mapreduce.lib.output.*;
import org.apache.hadoop.util.*;
import com.google.
public class topN extends Configured implements Tool {

  public static void main(String args[]) throws Exception {
    // Get input arguments and make sure they are sufficient
    Configuration conf = new Configuration();
    String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
    if (otherArgs.length != 3) {
    System.err.println("Usage: TopWord <in> <out> <N>");
    System.exit(3);
    }
    // Set up the driver class
    int res = ToolRunner.run(new topN(), args);
    System.exit(res);
  }

  public int run(String[] args) throws Exception {

    // Extract input and output path from cmdline args
    Path inputPath = new Path(args[0]);
    Path outputPath = new Path(args[1]);

    Configuration konf = getConf();

    // Create a JobConf Object
    Job job = new Job(konf, this.getClass().toString());

    // Read the paths as an HDFS complient path
    FileInputFormat.setInputPaths(job, inputPath);
    FileOutputFormat.setOutputPath(job, outputPath);

    // Create a Job name for reference
    job.setJobName("WordCount");
    // Tell MapReduce what to look for in a executabe jar file
    job.setJarByClass(topN.class);
    //
    job.setInputFormatClass(TextInputFormat.class);
    job.setOutputFormatClass(TextOutputFormat.class);
    // Tell Hadoop input/output key/value data type
    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(IntWritable.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(IntWritable.class);
    // Set mapper and reducer classes
    job.setMapperClass(Map.class);
    job.setCombinerClass(Reduce.class);
    job.setReducerClass(Reduce.class);
    job.setNumReduceTasks(3);
    // Return status to main
    return job.waitForCompletion(true) ? 0 : 1;
  }

  public static class Map extends Mapper<LongWritable, Text, Text, IntWritable> {

    // Do a Hashmap based grouping
    // Setup Hashmultiset
    private final HashMultiset<String> wordCount = HashMultiset.create();

    @Override
    public void map(LongWritable key, Text value,
                    Mapper.Context context) throws IOException, InterruptedException {

      String[] tokens = value.toString().split("\\s+");
      for (String token : tokens) {
        wordCount.add(token)
      }
    }
  }

  public static class Reduce extends Reducer<Text, IntWritable, Text, IntWritable> {

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
      Text key= new Text();
      LongWritable value = new LongWritable();
      for (Entry<String> entry: wordCountSet.entrySet()) {
        key.set(entry.getElement());
        value.set(entry.getCount());
        context.write(key, value);
      }
    }
  }
}
