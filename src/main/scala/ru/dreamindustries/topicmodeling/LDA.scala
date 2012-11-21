package ru.dreamindustries.topicmodeling

import breeze.stats.distributions.{Multinomial, DiscreteDistr}
import breeze.linalg.{DenseVector, SparseVector}

class LDA(ntopics: Int, theta: Seq[DiscreteDistr[Int]], phi: Seq[Topic]) {
}

object LDA {
  def topics(corpus: Seq[(Words, WordTopics)], ntopics: Int)(implicit totalWords: Int = Int.MaxValue): Seq[Topic] = {
    assert(corpus.forall(d => d._1.length == d._2.length), "topic annotations must correspond to words")

    val (words, topics) = corpus.unzip
    val W = words.flatten
    val Z = topics.flatten
    (0 to ntopics).map { k =>
      val topic = SparseVector.zeros[Double](totalWords)
      for (w <- Z.zipWithIndex.filter{case(z, _) => z == k}.unzip._2.map(i => W(i)))
        topic(w) += 1.0
      val normalize = Z.count(_ == k).toDouble
      topic :/= normalize
      Multinomial(topic)
    }
  }

  def topicProportions(topicAssignment: WordTopics, ntopics: Int) = {
    val theta = new DenseVector[Double]((0 to ntopics).map(k => topicAssignment.count(_ == k).toDouble).toArray)
    theta :/= theta.valuesIterator.reduce(_ + _)
    theta
  }
}

abstract case class NumberOfTopics()

sealed case class PredefinedNumberOfTopics(K: Int) extends NumberOfTopics()

sealed case class UnknownNumberOfTopics() extends NumberOfTopics()

trait LDAInference[TopicsNumber <: NumberOfTopics] extends ((Corpus, TopicsNumber) => LDA)
