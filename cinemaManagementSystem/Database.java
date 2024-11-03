package cinemaManagementSystem;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.Statement;

public class Database {
	
	private String user = "root";
	private String password = "";
	private String url = "jdbc:mysql://localhost/movieticketbookingsystem";
	private Statement statement;
	
	public Database() {
		try {
			Connection connection = DriverManager.getConnection(url, user, password);
			statement = connection.createStatement(ResultSet.TYPE_SCROLL_INSENSITIVE,
					ResultSet.CONCUR_READ_ONLY);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public Statement getStatement() {
		return statement;
	}

}
