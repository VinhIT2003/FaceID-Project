-- Tạo database
CREATE DATABASE Face_ID_Management;
GO

-- Sử dụng database vừa tạo
USE Face_ID_Management;
GO

-- Tạo bảng Employee
CREATE TABLE Employee (
    Emp_No VARCHAR(20) NOT NULL PRIMARY KEY,
    Name NVARCHAR(100) NOT NULL,
    Face_ID VARBINARY(MAX) NULL
);
GO
