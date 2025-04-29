import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

public class ImpalaJDBCExample {
    // JDBC connection details
    private static final String JDBC_URL = "jdbc:impala://<impala-host>:21050;AuthMech=0";
    private static final String DRIVER_CLASS = "com.cloudera.impala.jdbc.Driver";

    public static void main(String[] args) {
        Connection conn = null;
        Statement stmt = null;

        try {
            // Load the JDBC driver
            Class.forName(DRIVER_CLASS);

            // Establish connection
            conn = DriverManager.getConnection(JDBC_URL);
            stmt = conn.createStatement();

            // 1. Create a database
            String createDatabase = "CREATE DATABASE IF NOT EXISTS sample_db";
            stmt.executeUpdate(createDatabase);
            System.out.println("Database 'sample_db' created or already exists.");

            // 2. Use the database
            stmt.execute("USE sample_db");

            // 3. Create a table
            String createTable = "CREATE TABLE IF NOT EXISTS employees (" +
                                 "id INT, " +
                                 "name STRING, " +
                                 "salary DOUBLE) " +
                                 "STORED AS PARQUET";
            stmt.executeUpdate(createTable);
            System.out.println("Table 'employees' created or already exists.");

            // 4. Insert data
            String insertData = "INSERT INTO employees (id, name, salary) VALUES " +
                                "(1, 'Alice', 75000.0), " +
                                "(2, 'Bob', 80000.0), " +
                                "(3, 'Charlie', 65000.0)";
            stmt.executeUpdate(insertData);
            System.out.println("Data inserted into 'employees' table.");

            // 5. Run a simple query
            String query = "SELECT id, name, salary FROM employees WHERE salary > 70000";
            ResultSet rs = stmt.executeQuery(query);

            // Process query results
            System.out.println("Query results (employees with salary > 70000):");
            while (rs.next()) {
                int id = rs.getInt("id");
                String name = rs.getString("name");
                double salary = rs.getDouble("salary");
                System.out.printf("ID: %d, Name: %s, Salary: %.2f%n", id, name, salary);
            }

        } catch (ClassNotFoundException e) {
            System.err.println("JDBC Driver not found: " + e.getMessage());
        } catch (SQLException e) {
            System.err.println("SQL Error: " + e.getMessage());
        } finally {
            // Clean up resources
            try {
                if (stmt != null) stmt.close();
                if (conn != null) conn.close();
            } catch (SQLException e) {
                System.err.println("Error closing resources: " + e.getMessage());
            }
        }
    }
}