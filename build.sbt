name := "topicmodeling"

version := "0.01"

scalaVersion := "2.9.2"

libraryDependencies  ++= Seq(
  "org.scalanlp" %% "breeze-math" % "0.1",
  "org.scalanlp" %% "breeze-learn" % "0.1",
  "org.scalanlp" %% "breeze-process" % "0.1",
  "org.scalanlp" %% "breeze-viz" % "0.1"
)

resolvers ++= Seq(
  "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/"
)

