package com.example;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;

public class LogAnalysis {

    // Mapper class
    public static class LogMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
        private final static IntWritable one = new IntWritable(1);
        private Text keyOut = new Text();
        private SimpleDateFormat inputFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        private SimpleDateFormat hourFormat = new SimpleDateFormat("yyyy-MM-dd HH:00");

        @Override
        protected void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            String line = value.toString().trim();
            try {
                // Split log line: expected format "yyyy-MM-dd HH:mm:ss LEVEL message"
                String[] parts = line.split("\\s+", 3);
                if (parts.length < 3) return; // Skip malformed lines

                String timestamp = parts[0] + " " + parts[1];
                String logLevel = parts[2].split("\\s+")[0]; // Get log level (e.g., INFO, ERROR)

                // Validate log level
                if (!logLevel.matches("INFO|ERROR|WARN|DEBUG")) return;

                // Parse timestamp and extract hour
                Date date = inputFormat.parse(timestamp);
                String hour = hourFormat.format(date);

                // Set output key: "yyyy-MM-dd HH:00 LEVEL"
                keyOut.set(hour + " " + logLevel);
                context.write(keyOut, one);
            } catch (ParseException e) {
                // Skip lines with invalid timestamp
            }
        }
    }

    // Reducer class
    public static class LogReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        private IntWritable result = new IntWritable();

        @Override
        protected void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }

    // Driver class
    public static class LogAnalysisDriver {
        public static void main(String[] args) throws Exception {
            if (args.length != 2) {
                System.err.println("Usage: LogAnalysis <input path> <output path>");
                System.exit(-1);
            }

            Configuration conf = new Configuration();
            Job job = Job.getInstance(conf, "Log Analysis");

            job.setJarByClass(LogAnalysisDriver.class);
            job.setMapperClass(LogMapper.class);
            job.setReducerClass(LogReducer.class);

            job.setMapOutputKeyClass(Text.class);
            job.setMapOutputValueClass(IntWritable.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(IntWritable.class);

            FileInputFormat.addInputPath(job, new Path(args[0]));
            FileOutputFormat.setOutputPath(job, new Path(args[1]));

            System.exit(job.waitForCompletion(true) ? 0 : 1);
        }
    }
}