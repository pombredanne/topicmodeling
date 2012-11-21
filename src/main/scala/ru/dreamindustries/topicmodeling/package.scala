package ru.dreamindustries

import breeze.linalg.SparseVector
import breeze.stats.distributions.DiscreteDistr

package object topicmodeling {
  implicit def Int2NumberOfTopics(k: Int) = {
    new PredefinedNumberOfTopics(k)
  }

  type BagOfWords = SparseVector[Int]
  type Words = Seq[Int]

  type WordTopics = Seq[Int]

  type Corpus = Seq[Words]

  type Topic = DiscreteDistr[Int]

  def bagOfWords(document: Words)(implicit totalWords: Int = Int.MaxValue): BagOfWords = {
    val grouped = document.view.groupBy(x => x)
    val bag = SparseVector.zeros[Int](totalWords)

    bag.reserve(grouped.size)

    for ((word, usages) <- grouped)
      bag(word) = usages.length

    bag
  }
}
